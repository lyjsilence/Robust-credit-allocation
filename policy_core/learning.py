"""Alternating policy and robustness-parameter optimization."""

from dataclasses import dataclass
import copy
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from .evaluation import dro_value, ipw_value
from .models import (
    build_policy,
    build_propensity_model,
    set_trainable,
)


def project_alpha(log_alpha, args):
    """Optional upper projection of the positive robustness parameter."""
    upper = getattr(args, "alpha_upper_bound", None)
    if upper is None:
        return
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError("alpha_upper_bound must be finite and positive")
    with torch.no_grad():
        log_alpha.clamp_(max=upper + math.log(-math.expm1(-upper)))


@dataclass(frozen=True)
class Candidate:
    path: Path
    validation_value: float
    criterion: float
    alpha: Optional[float]


@dataclass
class JointTrainingCheckpoint:
    """Retain a policy-alpha pair using training joint-DRO only."""

    state_dict: dict
    log_alpha: torch.Tensor
    value: float
    consecutive_rejections: int = 0

    @staticmethod
    def _clone_state(policy):
        return {
            name: tensor.detach().clone()
            for name, tensor in policy.state_dict().items()
        }

    @classmethod
    def capture(cls, policy, log_alpha, value, path):
        checkpoint = cls(
            state_dict=cls._clone_state(policy),
            log_alpha=log_alpha.detach().clone(),
            value=float(value),
        )
        torch.save(checkpoint.state_dict, path)
        return checkpoint

    def restore(self, policy, log_alpha, path):
        policy.load_state_dict(self.state_dict)
        with torch.no_grad():
            log_alpha.copy_(self.log_alpha)
        torch.save(self.state_dict, path)

    def consider(self, policy, log_alpha, value, path, min_delta):
        value = float(value)
        if np.isfinite(value) and value > self.value + float(min_delta):
            self.state_dict = self._clone_state(policy)
            self.log_alpha = log_alpha.detach().clone()
            self.value = value
            self.consecutive_rejections = 0
            torch.save(self.state_dict, path)
            return True
        self.consecutive_rejections += 1
        self.restore(policy, log_alpha, path)
        return False


class PolicyObjective:
    def __init__(self, policy_data, profile, device):
        self.args = policy_data["args"]
        self.profile = profile
        self.device = device
        self.x = torch.from_numpy(policy_data["X"]).to(torch.float32).to(device)
        self.y = torch.from_numpy(policy_data["Y"]).to(torch.float32).to(device)
        self.action = torch.from_numpy(policy_data["T"]).to(torch.int).to(device)
        self.propensity = (
            torch.from_numpy(policy_data["prop_score"]).to(torch.float32).to(device)
        )
        self.checkpoint_x = (
            torch.from_numpy(policy_data.get("X_checkpoint", policy_data["X"]))
            .to(torch.float32)
            .to(device)
        )
        self.checkpoint_y = (
            torch.from_numpy(policy_data.get("Y_checkpoint", policy_data["Y"]))
            .to(torch.float32)
            .to(device)
        )
        self.checkpoint_action = (
            torch.from_numpy(policy_data.get("T_checkpoint", policy_data["T"]))
            .to(torch.int)
            .to(device)
        )
        self.checkpoint_propensity = (
            torch.from_numpy(
                policy_data.get("prop_score_checkpoint", policy_data["prop_score"])
            )
            .to(torch.float32)
            .to(device)
        )
        self.indicator_temperature = profile.indicator_temperature

    def set_indicator_temperature(self, temperature):
        self.indicator_temperature = float(temperature)

    def _weights(self, propensity_model, logits):
        propensity = self.propensity
        if self.args.propensity_met == "unknown":
            estimated = torch.softmax(propensity_model(self.x), dim=1)
            propensity = torch.clamp(
                estimated[torch.arange(estimated.size(0)), self.action.squeeze()],
                min=0.01,
            ).reshape(-1, 1)

        match = torch.exp(
            -F.cross_entropy(
                logits / self.indicator_temperature,
                self.action.squeeze().to(torch.int64),
                reduction="none",
            ).reshape(-1, 1)
        )
        normalizer = (
            match / propensity
            if self.args.estimator_choices == "normalized_estimator"
            else torch.ones(1, device=self.device)
        )
        return match, normalizer, propensity

    def policy_loss(self, propensity_model, alpha_history, logits):
        match, normalizer, propensity = self._weights(propensity_model, logits)
        if self.args.robust_met == "non-robust":
            samples = match * self.y / propensity
            return -torch.mean(samples) / torch.mean(normalizer)

        log_alpha = alpha_history[-1]
        log_alpha.requires_grad_(False)
        alpha = F.softplus(log_alpha)
        samples = match * torch.exp(-self.y / alpha) / propensity
        return torch.mean(samples) / torch.mean(normalizer)

    def nonrobust_policy_loss(self, propensity_model, logits):
        """Smooth teacher-independent warm-up using the observed IPW value."""
        match, normalizer, propensity = self._weights(propensity_model, logits)
        samples = match * self.y / propensity
        return -torch.mean(samples) / torch.mean(normalizer)

    def weighted_classification_loss(
        self, propensity_model, logits, use_checkpoint=False
    ):
        """Fisher-consistent IPW classification surrogate for warm-up."""
        if use_checkpoint:
            x = self.checkpoint_x
            y = self.checkpoint_y
            action = self.checkpoint_action
            propensity = self.checkpoint_propensity
        else:
            x = self.x
            y = self.y
            action = self.action
            propensity = self.propensity
        if self.args.propensity_met == "unknown":
            estimated = torch.softmax(propensity_model(x), dim=1)
            propensity = torch.clamp(
                estimated[
                    torch.arange(estimated.size(0)), action.squeeze()
                ],
                min=0.01,
            ).reshape(-1, 1)
        weights = torch.clamp(y / propensity, min=0.0)
        losses = F.cross_entropy(
            logits,
            action.squeeze().to(torch.int64),
            reduction="none",
        ).reshape(-1, 1)
        return torch.mean(weights * losses) / torch.clamp(
            torch.mean(weights), min=1e-8
        )

    def alpha_loss(self, propensity_model, log_alpha, logits):
        match, normalizer, propensity = self._weights(propensity_model, logits)
        with torch.enable_grad():
            alpha = F.softplus(log_alpha)
            if (getattr(self.args, "alpha_upper_bound", None) is not None
                    and self.args.estimator_choices == "unnormalized_estimator"):
                log_match = -F.cross_entropy(
                    logits / self.indicator_temperature,
                    self.action.squeeze().long(), reduction="none"
                ).reshape(-1, 1)
                terms = log_match - self.y / alpha - torch.log(propensity)
                log_mean = torch.logsumexp(terms.reshape(-1), 0) - math.log(terms.numel())
                return alpha * (log_mean + self.args.amb_radius)
            samples = match * torch.exp(-self.y / alpha) / propensity
            return (
                alpha * torch.log(torch.mean(samples) / torch.mean(normalizer))
                + alpha * self.args.amb_radius
            )


def validation_tensors(policy_data, device):
    return (
        torch.tensor(policy_data["X_val"]).to(device),
        torch.tensor(policy_data["T_val"]).to(device),
        torch.tensor(policy_data["Y_val"]).to(device),
        torch.tensor(policy_data["prop_score_val"]).to(device),
    )


def validation_value(policy, policy_data, log_alpha, profile, device):
    x, action, outcome, propensity = validation_tensors(policy_data, device)
    if profile.validation_matching == "soft" or policy_data["robust_met"] == "robust":
        value, _ = dro_value(
            policy,
            x,
            action,
            outcome,
            propensity,
            log_alpha,
            policy_data["args"],
            matching=profile.validation_matching,
        )
        return value
    return ipw_value(
        policy,
        x,
        action,
        outcome,
        propensity,
        policy_data["args"],
        matching=profile.validation_matching,
    )


def _temperature(args, profile, epoch, anneal):
    endpoint = float(
        getattr(args, "policy_temperature_end", profile.indicator_temperature)
    )
    if not anneal:
        return endpoint
    start = float(getattr(args, "policy_temperature_start", endpoint))
    anneal_epochs = max(int(getattr(args, "policy_temperature_anneal_epochs", 1)), 1)
    fraction = min(epoch / anneal_epochs, 1.0)
    if start <= 0 or endpoint <= 0:
        raise ValueError("Policy matching temperatures must be positive")
    return start * (endpoint / start) ** fraction


def _training_loss(loss_function, policy, x):
    with torch.no_grad():
        return float(loss_function(policy(x)).item())


def optimize_policy(
    loss_function,
    policy_data,
    policy,
    path,
    profile,
    device,
    log_alpha,
    objective=None,
    anneal_temperature=False,
    maximum_epochs=None,
    minimum_epochs_override=None,
    checkpoint_loss_function=None,
    checkpoint_x=None,
):
    args = policy_data["args"]
    optimizer = Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stale_steps = 0
    x = torch.from_numpy(policy_data["X"]).to(torch.float32).to(device)
    criterion_x = x if checkpoint_x is None else checkpoint_x
    criterion_loss_function = (
        loss_function
        if checkpoint_loss_function is None
        else checkpoint_loss_function
    )

    if profile.checkpoint == "final":
        previous_value = torch.inf
        for _ in range(args.optimization_epochs):
            optimizer.zero_grad()
            loss = loss_function(policy(x))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                value = validation_value(policy, policy_data, log_alpha, profile, device)
                difference = previous_value - value
                if difference < 0.001 or value.isnan():
                    stale_steps += 1
                    if stale_steps >= 10:
                        break
                else:
                    previous_value = value
                    stale_steps = 0
        torch.save(policy.state_dict(), path)
        return loss.item(), value.item(), {
            "epochs": _ + 1,
            "training_loss": None,
        }

    training_checkpoint = bool(getattr(args, "training_only_checkpoint", False))
    default_minimum = (
        profile.minimum_nonrobust_epochs
        if policy_data["robust_met"] == "non-robust"
        else 0
    )
    minimum_epochs = int(
        minimum_epochs_override
        if minimum_epochs_override is not None
        else getattr(args, "policy_minimum_epochs", default_minimum)
    )
    total_epochs = int(
        args.optimization_epochs if maximum_epochs is None else maximum_epochs
    )
    patience = int(getattr(args, "policy_patience", 10))
    min_delta = float(getattr(args, "policy_min_delta", 0.001))
    endpoint = float(
        getattr(args, "policy_temperature_end", profile.indicator_temperature)
    )
    if objective is not None:
        objective.set_indicator_temperature(endpoint)
    if training_checkpoint:
        best_criterion = _training_loss(
            criterion_loss_function, policy, criterion_x
        )
        best_value = validation_value(policy, policy_data, log_alpha, profile, device)
    else:
        with torch.no_grad():
            best_value = validation_value(policy, policy_data, log_alpha, profile, device)
        best_criterion = -float(best_value.item())
    torch.save(policy.state_dict(), path)
    for epoch in range(total_epochs):
        if objective is not None:
            objective.set_indicator_temperature(
                _temperature(args, profile, epoch, anneal_temperature)
            )
        optimizer.zero_grad()
        loss = loss_function(policy(x))
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if objective is not None:
                objective.set_indicator_temperature(endpoint)
            if training_checkpoint:
                criterion = _training_loss(
                    criterion_loss_function, policy, criterion_x
                )
                improved = criterion < best_criterion - min_delta
                value = None
            else:
                value = validation_value(policy, policy_data, log_alpha, profile, device)
                criterion = -float(value.item())
                improved = criterion < best_criterion - min_delta
            if improved:
                best_criterion = criterion
                if value is not None:
                    best_value = value
                stale_steps = 0
                torch.save(policy.state_dict(), path)
            else:
                stale_steps += 1
                anneal_epochs = (
                    int(getattr(args, "policy_temperature_anneal_epochs", 0))
                    if anneal_temperature
                    else 0
                )
                if (
                    stale_steps >= patience
                    and epoch + 1 >= minimum_epochs
                    and epoch + 1 >= anneal_epochs
                ):
                    break
    policy.load_state_dict(torch.load(path))
    if objective is not None:
        objective.set_indicator_temperature(endpoint)
    if training_checkpoint:
        with torch.no_grad():
            best_value = validation_value(
                policy, policy_data, log_alpha, profile, device
            )
    return loss.item(), best_value.item(), {
        "epochs": epoch + 1,
        "training_loss": best_criterion if training_checkpoint else None,
    }


def optimize_alpha(loss_function, policy_data, alpha_history, policy, profile, device):
    args = policy_data["args"]
    log_alpha = alpha_history[-1]
    project_alpha(log_alpha, args)
    log_alpha.requires_grad_(True)
    optimizer = Adam([log_alpha], lr=args.lr, weight_decay=args.weight_decay)
    stale_steps = 0

    if profile.restore_best_alpha and getattr(args, "training_only_checkpoint", False):
        best_loss = float(loss_function(log_alpha).item())
        best_alpha = log_alpha.detach().clone()
        patience = int(getattr(args, "alpha_patience", 25))
        minimum_epochs = int(getattr(args, "alpha_minimum_epochs", 25))
        min_delta = float(getattr(args, "alpha_min_delta", 1e-6))
        for epoch in range(args.optimization_epochs):
            optimizer.zero_grad()
            loss = loss_function(log_alpha)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            project_alpha(log_alpha, args)
            with torch.no_grad():
                current_loss = float(loss_function(log_alpha).item())
                if current_loss < best_loss - min_delta:
                    best_loss = current_loss
                    best_alpha = log_alpha.detach().clone()
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= patience and epoch + 1 >= minimum_epochs:
                        break
        with torch.no_grad():
            log_alpha.copy_(best_alpha)
        project_alpha(log_alpha, args)
        return log_alpha

    if profile.restore_best_alpha:
        best_value = -torch.inf
        best_alpha = log_alpha.detach().clone()
        for _ in range(args.optimization_epochs):
            optimizer.zero_grad()
            loss = loss_function(log_alpha)
            loss.backward()
            optimizer.step()
            project_alpha(log_alpha, args)
            with torch.no_grad():
                value = validation_value(policy, policy_data, log_alpha, profile, device)
                if value > best_value + 0.001:
                    best_value = value
                    best_alpha = log_alpha.detach().clone()
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= 10:
                        break
        with torch.no_grad():
            log_alpha.copy_(best_alpha)
        project_alpha(log_alpha, args)
        return log_alpha

    previous_value = torch.inf
    for _ in range(args.optimization_epochs):
        optimizer.zero_grad()
        loss = loss_function(log_alpha)
        loss.backward()
        optimizer.step()
        project_alpha(log_alpha, args)
        with torch.no_grad():
            value = validation_value(policy, policy_data, log_alpha, profile, device)
            difference = previous_value - value
            if difference < 0.001 or value.isnan():
                stale_steps += 1
                if stale_steps >= 10:
                    break
            else:
                previous_value = value
                stale_steps = 0
    return log_alpha


def joint_training_dro_value(policy, objective, log_alpha, args, profile):
    """Evaluate joint DRO on the frozen training-internal checkpoint sample."""
    endpoint = float(
        getattr(args, "policy_temperature_end", profile.indicator_temperature)
    )
    with torch.no_grad():
        value, _ = dro_value(
            policy,
            objective.checkpoint_x,
            objective.checkpoint_action,
            objective.checkpoint_y,
            objective.checkpoint_propensity,
            log_alpha,
            args,
            matching="soft",
            temperature=endpoint,
            radius=args.amb_radius,
        )
    return float(value.item())


def audit_stage_training_q(policy, objective, log_alpha, args):
    """Observe a retained stage pair; never select or optimize from this value.

    Uses the frozen training-internal checkpoint sample, hard matching and the
    normalized estimator at the current alpha (no alpha refit). This samples
    stage boundaries, not every epoch. A private policy copy protects buffers
    and module modes; global Python, NumPy and Torch RNG streams are restored.
    Undefined hard estimates (e.g. zero matches) remain nonfinite and flagged.
    """
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            observed_policy = copy.deepcopy(policy)
            observed_policy.eval()
            evaluation_args = copy.copy(args)
            evaluation_args.estimator_choices = "normalized_estimator"
            value, _ = dro_value(
                observed_policy,
                objective.checkpoint_x,
                objective.checkpoint_action,
                objective.checkpoint_y,
                objective.checkpoint_propensity,
                log_alpha.detach().clone(),
                evaluation_args,
                matching="hard",
                radius=args.amb_radius,
            )
            value = float(value.item())
            return {
                "audit_stage_training_q": value,
                "audit_stage_alpha": float(F.softplus(log_alpha).item()),
                "audit_stage_training_q_finite": bool(np.isfinite(value)),
                "audit_sample": "training_internal_checkpoint",
                "audit_sample_size": int(objective.checkpoint_x.shape[0]),
                "audit_matching": "hard",
                "audit_estimator": "normalized_estimator",
                "audit_radius": float(args.amb_radius),
                "audit_scope": "retained_stage_boundaries_only",
            }
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def fit_candidate(
    policy_data,
    depth,
    replication,
    path,
    profile,
    device,
    return_diagnostics=False,
    initial_policy_state=None,
):
    args = policy_data["args"]
    objective = PolicyObjective(policy_data, profile, device)
    policy = build_policy(policy_data, depth, args.baseline, profile, device)
    if initial_policy_state is not None:
        policy.load_state_dict(initial_policy_state)
    log_alpha = nn.Parameter(torch.tensor(1.0))
    project_alpha(log_alpha, args)
    alpha_history = [log_alpha]
    alpha_values = [F.softplus(log_alpha).item()]
    propensity_model = build_propensity_model(args, policy_data, profile, device)
    iterations = args.max_iter if policy_data["robust_met"] == "robust" else 1
    validation = None
    optimization_history = []
    checkpoint_strategy = getattr(args, "policy_checkpoint_strategy", "stage_loss")
    if checkpoint_strategy not in ("stage_loss", "joint_training_dro"):
        raise ValueError(f"Unknown policy checkpoint strategy: {checkpoint_strategy}")

    pretrain_epochs = int(getattr(args, "policy_pretrain_epochs", 0))
    if policy_data["robust_met"] == "robust" and pretrain_epochs > 0:
        set_trainable(policy, True)
        pretrain_objective = getattr(
            args, "policy_pretrain_objective", "weighted_classification"
        )
        if pretrain_objective == "weighted_classification":
            pretrain_loss = lambda logits: objective.weighted_classification_loss(
                propensity_model, logits
            )
        elif pretrain_objective == "nonrobust_value":
            pretrain_loss = lambda logits: objective.nonrobust_policy_loss(
                propensity_model, logits
            )
        else:
            raise ValueError(
                f"Unknown policy pretrain objective: {pretrain_objective}"
            )
        pretrain_checkpoint_split = getattr(
            args, "policy_pretrain_checkpoint_split", "fit"
        )
        if pretrain_checkpoint_split not in ("fit", "optimization_holdout"):
            raise ValueError(
                "Unknown policy pretrain checkpoint split: "
                f"{pretrain_checkpoint_split}"
            )
        checkpoint_loss_function = None
        checkpoint_x = None
        if pretrain_checkpoint_split == "optimization_holdout":
            if pretrain_objective != "weighted_classification":
                raise ValueError(
                    "optimization-holdout pretrain checkpointing currently "
                    "requires weighted_classification"
                )
            checkpoint_loss_function = lambda logits: (
                objective.weighted_classification_loss(
                    propensity_model, logits, use_checkpoint=True
                )
            )
            checkpoint_x = objective.checkpoint_x
        _, validation, pretrain_stats = optimize_policy(
            pretrain_loss,
            policy_data,
            policy,
            path,
            profile,
            device,
            log_alpha,
            objective=objective,
            anneal_temperature=True,
            maximum_epochs=pretrain_epochs,
            minimum_epochs_override=min(
                int(getattr(args, "policy_pretrain_minimum_epochs", 100)),
                pretrain_epochs,
            ),
            checkpoint_loss_function=checkpoint_loss_function,
            checkpoint_x=checkpoint_x,
        )
        optimization_history.append(
            {
                "stage": pretrain_objective,
                "checkpoint_split": pretrain_checkpoint_split,
                **pretrain_stats,
            }
        )
        if getattr(args, "audit_stage_training_q", False):
            optimization_history[-1].update(
                audit_stage_training_q(policy, objective, log_alpha, args)
            )

    joint_checkpoint = None
    if (
        policy_data["robust_met"] == "robust"
        and checkpoint_strategy == "joint_training_dro"
    ):
        if not Path(path).is_file():
            torch.save(policy.state_dict(), path)
        policy.load_state_dict(torch.load(path))
        policy.eval()
        set_trainable(policy, False)
        logits = policy(objective.x)
        alpha_loss = lambda alpha: objective.alpha_loss(
            propensity_model, alpha, logits
        )
        optimize_alpha(
            alpha_loss, policy_data, alpha_history, policy, profile, device
        )
        alpha_values.append(F.softplus(log_alpha).item())
        initial_joint_value = joint_training_dro_value(
            policy, objective, log_alpha, args, profile
        )
        joint_checkpoint = JointTrainingCheckpoint.capture(
            policy, log_alpha, initial_joint_value, path
        )
        optimization_history.append(
            {
                "stage": "joint_training_dro_initial",
                "joint_training_dro": initial_joint_value,
                "alpha": alpha_values[-1],
                "accepted": True,
            }
        )

    for iteration in range(iterations):
        if iteration > 0:
            policy.load_state_dict(torch.load(path))
        set_trainable(policy, True)
        policy_loss = lambda logits: objective.policy_loss(
            propensity_model, alpha_history, logits
        )
        _, validation, policy_stats = optimize_policy(
            policy_loss,
            policy_data,
            policy,
            path,
            profile,
            device,
            log_alpha,
            objective=objective,
            anneal_temperature=iteration == 0 and pretrain_epochs == 0,
        )
        optimization_history.append(
            {"stage": f"robust_{iteration}", **policy_stats}
        )

        if policy_data["robust_met"] == "robust":
            policy.load_state_dict(torch.load(path))
            policy.eval()
            set_trainable(policy, False)
            logits = policy(objective.x)
            alpha_loss = lambda alpha: objective.alpha_loss(
                propensity_model, alpha, logits
            )
            new_log_alpha = optimize_alpha(
                alpha_loss, policy_data, alpha_history, policy, profile, device
            )
            attempted_alpha = F.softplus(new_log_alpha).item()
            stage_record = optimization_history[-1]
            if joint_checkpoint is not None:
                joint_value = joint_training_dro_value(
                    policy, objective, log_alpha, args, profile
                )
                accepted = joint_checkpoint.consider(
                    policy,
                    log_alpha,
                    joint_value,
                    path,
                    min_delta=float(
                        getattr(args, "joint_dro_min_delta", 1e-5)
                    ),
                )
                retained_alpha = F.softplus(log_alpha).item()
                alpha_values.append(retained_alpha)
                optimization_history.append(
                    {
                        "stage": f"joint_training_dro_{iteration}",
                        "joint_training_dro": joint_value,
                        "attempted_alpha": attempted_alpha,
                        "retained_alpha": retained_alpha,
                        "accepted": accepted,
                        "consecutive_rejections": (
                            joint_checkpoint.consecutive_rejections
                        ),
                    }
                )
                if getattr(args, "audit_stage_training_q", False):
                    stage_record.update(
                        audit_stage_training_q(policy, objective, log_alpha, args)
                    )
                if (
                    joint_checkpoint.consecutive_rejections
                    >= int(getattr(args, "joint_dro_patience", 2))
                ):
                    break
            else:
                alpha_values.append(attempted_alpha)
                if getattr(args, "audit_stage_training_q", False):
                    stage_record.update(
                        audit_stage_training_q(policy, objective, log_alpha, args)
                    )
            if (
                joint_checkpoint is None
                and abs(alpha_values[-1] - alpha_values[-2]) < 0.001
            ):
                break

    policy.load_state_dict(torch.load(path))
    policy.eval()
    training_value = joint_training_dro_value(
        policy, objective, log_alpha, args, profile
    )
    with torch.no_grad():
        validation = validation_value(
            policy, policy_data, log_alpha, profile, device
        ).item()
    if return_diagnostics:
        return (
            Path(path),
            validation,
            alpha_values,
            training_value,
            optimization_history,
        )
    return Path(path), validation, alpha_values
