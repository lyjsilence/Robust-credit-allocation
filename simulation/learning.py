"""Alternating policy and robustness-parameter optimization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from evaluation import dro_value, ipw_value
from models import PropensityModel, build_policy, set_trainable


@dataclass(frozen=True)
class Candidate:
    path: Path
    validation_value: float
    criterion: float
    alpha: Optional[float]


class PolicyObjective:
    """Training objective with the exact normalized IPW calculations."""

    def __init__(self, policy_data, device):
        self.device = device
        self.args = policy_data["args"]
        self.propensity_method = policy_data["propensity_met"]
        self.estimator = policy_data["estimator_choices"]
        self.radius = policy_data["amb_radius"]
        self.x = torch.from_numpy(policy_data["X"]).to(torch.float32).to(device)
        self.y = torch.from_numpy(policy_data["Y"]).to(torch.float32).to(device)
        self.action = torch.from_numpy(policy_data["T"]).to(torch.int).to(device)
        self.propensity = (
            torch.from_numpy(policy_data["prop_score"]).to(torch.float32).to(device)
        )

    def _weights(self, propensity_model, logits):
        propensity = self.propensity
        if self.propensity_method == "unknown":
            estimated = torch.softmax(propensity_model(self.x), dim=1)
            propensity = torch.clamp(
                estimated[torch.arange(estimated.size(0)), self.action.squeeze()],
                min=0.01,
            ).reshape(-1, 1)

        match = torch.exp(
            -F.cross_entropy(
                logits / 0.05,
                self.action.squeeze().to(torch.int64),
                reduction="none",
            ).reshape(-1, 1)
        )
        normalizer = (
            match / propensity
            if self.estimator == "normalized_estimator"
            else torch.Tensor([1]).to(self.device)
        )
        return match, normalizer, propensity

    def policy_loss(self, propensity_model, robust_method, alpha_history, logits):
        match, normalizer, propensity = self._weights(propensity_model, logits)
        if robust_method == "non-robust":
            samples = match * self.y / propensity
            return -torch.mean(samples) / torch.mean(normalizer)

        log_alpha = alpha_history[-1]
        log_alpha.requires_grad_(False)
        alpha = F.softplus(log_alpha)
        samples = match * torch.exp(-self.y / alpha) / propensity
        return torch.mean(samples) / torch.mean(normalizer)

    def alpha_loss(self, propensity_model, log_alpha, logits):
        match, normalizer, propensity = self._weights(propensity_model, logits)
        with torch.enable_grad():
            alpha = F.softplus(log_alpha)
            samples = match * torch.exp(-self.y / alpha) / propensity
            return (
                alpha * torch.log(torch.mean(samples) / torch.mean(normalizer))
                + alpha * self.radius
            )


def validation_tensors(policy_data, device):
    return (
        torch.tensor(policy_data["X_val"]).to(device),
        torch.tensor(policy_data["T_val"]).to(device),
        torch.tensor(policy_data["Y_val"]).to(device),
        torch.tensor(policy_data["prop_score_val"]).to(device),
    )


def optimize_policy(loss_function, policy_data, policy, path, device, iteration, log_alpha):
    args = policy_data["args"]
    optimizer = Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value = -torch.inf
    stale_steps = 0

    for epoch in range(args.optimization_epochs):
        optimizer.zero_grad()
        loss = loss_function(policy(torch.from_numpy(policy_data["X"]).to(torch.float32).to(device)))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x, action, outcome, propensity = validation_tensors(policy_data, device)
            if policy_data["robust_met"] == "robust":
                value, _ = dro_value(
                    policy, action=action, outcome=outcome, propensity=propensity,
                    x=x, log_alpha=log_alpha, args=args
                )
            else:
                value = ipw_value(policy, x, action, outcome, propensity, args)

            if value > best_value + 0.001:
                best_value = value
                stale_steps = 0
                torch.save(policy.state_dict(), path)
            else:
                stale_steps += 1
                minimum_epochs = 100 if policy_data["robust_met"] == "non-robust" else 0
                if stale_steps >= 10 and epoch + 1 >= minimum_epochs:
                    break

    policy.load_state_dict(torch.load(path))
    return loss.item(), best_value.item()


def optimize_alpha(loss_function, policy_data, alpha_history, device, policy):
    args = policy_data["args"]
    log_alpha = alpha_history[-1]
    log_alpha.requires_grad_(True)
    optimizer = Adam([log_alpha], lr=args.lr, weight_decay=args.weight_decay)
    best_value = -torch.inf
    best_alpha = log_alpha.detach().clone()
    stale_steps = 0

    for _ in range(args.optimization_epochs):
        optimizer.zero_grad()
        loss = loss_function(log_alpha)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x, action, outcome, propensity = validation_tensors(policy_data, device)
            value, _ = dro_value(
                policy, x, action, outcome, propensity, log_alpha, args
            )
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
    return log_alpha


def fit_candidate(policy_data, depth, replication, device):
    args = policy_data["args"]
    objective = PolicyObjective(policy_data, device)
    policy = build_policy(policy_data, depth, args.baseline, device)
    log_alpha = nn.Parameter(torch.tensor(1.0))
    alpha_history = [log_alpha]
    alpha_values = [F.softplus(log_alpha).item()]

    if args.baseline == "None":
        filename = (
            f"opt_pol_{args.mode}_dist_{args.dist_shift}_{args.opt_procedure}"
            f"-exp_id_{replication}_{depth}.pth"
        )
    else:
        filename = (
            f"opt_pol_{args.mode}_dist_{args.dist_shift}_{args.opt_procedure}_"
            f"{args.baseline}-exp_id_{replication}_{depth}.pth"
        )
    path = Path(args.model_dir) / filename

    propensity_model = PropensityModel(
        args, policy_data["dim_X"], policy_data["num_treat"], device
    )
    iterations = args.max_iter if policy_data["robust_met"] == "robust" else 1
    validation_value = None

    for iteration in range(iterations):
        if iteration > 0:
            policy.load_state_dict(torch.load(path))

        set_trainable(policy, True)
        policy_loss = lambda logits: objective.policy_loss(
            propensity_model, policy_data["robust_met"], alpha_history, logits
        )
        _, validation_value = optimize_policy(
            policy_loss, policy_data, policy, path, device, iteration, log_alpha
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
                alpha_loss, policy_data, alpha_history, device, policy
            )
            alpha_values.append(F.softplus(new_log_alpha).item())
            if abs(alpha_values[-1] - alpha_values[-2]) < 0.001:
                break

    return path, validation_value, alpha_values
