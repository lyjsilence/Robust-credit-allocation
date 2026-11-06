"""Policy learning with known propensities and training-loss checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam


TRAINING_OPTIONS = {
    "learning_rate": 0.005,
    "weight_decay": 0.0,
    "max_epochs": 2000,
    "min_policy_epochs": 200,
    "policy_patience": 100,
    "policy_tolerance": 1e-5,
    "temperature": 0.05,
    "min_dual_epochs": 25,
    "dual_patience": 25,
    "dual_tolerance": 1e-6,
    "max_alternations": 5,
    "radius": 0.2,
}


class Policy(nn.Module):
    def __init__(self, width, input_dim=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim, width),
                nn.Linear(width, width),
                nn.Linear(width, 2),
            ]
        )

    def forward(self, x):
        x = self.layers[0](x).relu()
        x = self.layers[1](x).relu()
        return self.layers[2](x)


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


def fit_candidate(train, width, method, path, device):
    settings = TRAINING_OPTIONS
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

    return history
