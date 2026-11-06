"""Policy learning with known propensities and training-loss checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from .evaluation import dro_value, ipw_value
from .models import Policy


class PolicyObjective:
    def __init__(self, data, temperature, radius, device):
        self.x = torch.from_numpy(data["X"]).float().to(device)
        self.y = torch.from_numpy(data["Y"]).float().to(device)
        self.action = torch.from_numpy(data["T"]).int().to(device)
        self.propensity = torch.from_numpy(data["prop_score"]).float().to(device)
        self.temperature = temperature
        self.radius = radius

    def weights(self, logits):
        match = torch.exp(
            -F.cross_entropy(
                logits / self.temperature,
                self.action.squeeze().long(),
                reduction="none",
            ).reshape(-1, 1)
        )
        return match, match / self.propensity

    def policy_loss(self, logits, log_alpha, robust):
        match, normalizer = self.weights(logits)
        if not robust:
            samples = match * self.y / self.propensity
            return -samples.mean() / normalizer.mean()
        log_alpha.requires_grad_(False)
        alpha = F.softplus(log_alpha)
        samples = match * torch.exp(-self.y / alpha) / self.propensity
        return samples.mean() / normalizer.mean()

    def classification_loss(self, logits):
        weights = torch.clamp(self.y / self.propensity, min=0.0)
        losses = F.cross_entropy(
            logits, self.action.squeeze().long(), reduction="none"
        ).reshape(-1, 1)
        return (weights * losses).mean() / weights.mean().clamp_min(1e-8)

    def alpha_loss(self, logits, log_alpha):
        match, normalizer = self.weights(logits)
        alpha = F.softplus(log_alpha)
        samples = match * torch.exp(-self.y / alpha) / self.propensity
        return (
            alpha * torch.log(samples.mean() / normalizer.mean()) + alpha * self.radius
        )


def optimize_policy(policy, x, loss_function, path, settings):
    """Retain the lowest fitting loss; validation never selects an epoch."""
    optimizer = Adam(
        policy.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    with torch.no_grad():
        best_loss = loss_function(policy(x)).item()
    torch.save(policy.state_dict(), path)
    stale = 0
    for epoch in range(settings["max_epochs"]):
        optimizer.zero_grad()
        loss = loss_function(policy(x))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite policy loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = loss_function(policy(x)).item()
        if current < best_loss - settings["policy_tolerance"]:
            best_loss = current
            stale = 0
            torch.save(policy.state_dict(), path)
        else:
            stale += 1
            if (
                stale >= settings["policy_patience"]
                and epoch + 1 >= settings["min_policy_epochs"]
            ):
                break
    policy.load_state_dict(torch.load(path, weights_only=True))
    return {"epochs": epoch + 1, "training_loss": best_loss}


def optimize_alpha(log_alpha, loss_function, settings):
    log_alpha.requires_grad_(True)
    optimizer = Adam(
        [log_alpha], lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    best_loss = loss_function(log_alpha).item()
    best_alpha = log_alpha.detach().clone()
    stale = 0
    for epoch in range(settings["max_epochs"]):
        optimizer.zero_grad()
        loss = loss_function(log_alpha)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite dual loss")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            current = loss_function(log_alpha).item()
            if current < best_loss - settings["dual_tolerance"]:
                best_loss = current
                best_alpha = log_alpha.detach().clone()
                stale = 0
            else:
                stale += 1
                if (
                    stale >= settings["dual_patience"]
                    and epoch + 1 >= settings["min_dual_epochs"]
                ):
                    break
    with torch.no_grad():
        log_alpha.copy_(best_alpha)


def fit_candidate(train, validation, width, method, settings, path, device):
    objective = PolicyObjective(
        train, settings["temperature"], settings["radius"], device
    )
    policy = Policy(width, train["X"].shape[1]).to(device)
    log_alpha = nn.Parameter(torch.tensor(1.0))
    alphas = [F.softplus(log_alpha).item()]
    robust = method == "robust"
    history = []

    if robust:
        stats = optimize_policy(
            policy, objective.x, objective.classification_loss, path, settings
        )
        history.append({"stage": "weighted_classification", **stats})

    for iteration in range(settings["max_alternations"] if robust else 1):
        policy.requires_grad_(True)
        loss_function = lambda logits: objective.policy_loss(logits, log_alpha, robust)
        stats = optimize_policy(policy, objective.x, loss_function, path, settings)
        history.append({"stage": method, "iteration": iteration, **stats})
        if robust:
            policy.eval().requires_grad_(False)
            logits = policy(objective.x)
            optimize_alpha(
                log_alpha, lambda alpha: objective.alpha_loss(logits, alpha), settings
            )
            alphas.append(F.softplus(log_alpha).item())
            if abs(alphas[-1] - alphas[-2]) < 0.001:
                break

    policy.eval()
    with torch.no_grad():
        tensors = [
            torch.tensor(validation[key]).to(device)
            for key in ("X", "T", "Y", "prop_score")
        ]
        if robust:
            value = dro_value(policy, *tensors, log_alpha, settings["radius"])
        else:
            value = ipw_value(policy, *tensors)
        training_value = dro_value(
            policy,
            objective.x,
            objective.action,
            objective.y,
            objective.propensity,
            log_alpha,
            settings["radius"],
            temperature=settings["temperature"],
        )
    return {
        "validation_value": value.item(),
        "training_robust_value": training_value.item(),
        "alphas": alphas,
        "history": history,
    }
