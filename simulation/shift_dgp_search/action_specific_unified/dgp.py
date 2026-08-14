"""One simulation family with smooth and lower-tail outcome components."""

from dataclasses import asdict, dataclass

import numpy as np


SEVERITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SHIFT_SCALE = 0.20


@dataclass(frozen=True)
class DGPConfig:
    smooth_effect: float = 1.5
    smooth_sharpness: float = 6.0
    smooth_noise: float = 0.3
    bad_outcome: float = 0.0
    good_outcome: float = 0.90
    stable_outcome: float = 0.79
    risky_propensity: float = 0.75
    risk_low: float = 0.02
    risk_high: float = 0.78
    risk_threshold: float = 0.70
    risk_sharpness: float = 30.0


CONFIG = DGPConfig()


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def sample_uniform_ball(n, d, rng):
    direction = rng.normal(size=(n, d))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    return direction * rng.uniform(size=(n, 1)) ** (1.0 / d)


def sample_actions(probabilities, rng):
    return np.asarray(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=np.int64,
    )


def nonlinear_boundary(x2):
    boundary = np.empty_like(x2, dtype=np.float64)
    upper = x2 >= 0
    boundary[upper] = np.sqrt(np.maximum(0.25 - (x2[upper] - 0.5) ** 2, 0.0))
    boundary[~upper] = -np.sqrt(np.maximum(0.25 - (x2[~upper] + 0.5) ** 2, 0.0))
    return boundary


def decision_score(x, boundary):
    if boundary == "linear":
        return x[:, 0] - 0.8 * x[:, 1]
    if boundary == "nonlinear":
        return x[:, 0] - nonlinear_boundary(x[:, 1])
    raise ValueError("boundary must be 'linear' or 'nonlinear'")


def propensity_from_action(action, preferred_probability):
    propensity = np.full((action.size, 2), 1.0 - preferred_probability)
    propensity[np.arange(action.size), action] = preferred_probability
    return propensity


def pack_dataset(x, action, outcomes, propensity, mean, **extra):
    rows = np.arange(x.shape[0])
    data = {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1),
        "Y": outcomes[rows, action].reshape(-1, 1).astype(np.float32),
        "prop_score": propensity[rows, action].reshape(-1, 1).astype(np.float32),
        "Y_all": outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": mean.astype(np.float32),
    }
    data.update(extra)
    return data


def simulate_smooth_dataset(n, seed, boundary, config=CONFIG):
    """Nominal outcome component used by Tables 1 and 3 and the figures."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    score = decision_score(x, boundary)
    contrast = config.smooth_effect * np.tanh(config.smooth_sharpness * score)
    latent_mean = np.column_stack([-contrast, contrast])
    outcomes = sigmoid(latent_mean + rng.normal(size=(n, 2)) * config.smooth_noise)
    optimal = np.argmax(latent_mean, axis=1)
    propensity = propensity_from_action(optimal, config.risky_propensity)
    action = sample_actions(propensity, rng)
    return pack_dataset(
        x, action, outcomes, propensity, latent_mean,
        score=score.reshape(-1, 1).astype(np.float32),
    )


def risk_score(x, config=CONFIG):
    wave = np.sin(2.0 * np.pi * x[:, 2]) * np.sin(2.0 * np.pi * x[:, 3])
    return sigmoid(config.risk_sharpness * (wave - config.risk_threshold))


def simulate_tail_dataset(n, seed, config=CONFIG):
    """Lower-tail component used by both Table 2 perturbation modes."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    score = decision_score(x, "nonlinear")
    risky_action = (score >= 0).astype(np.int64)
    probability = config.risk_low + (config.risk_high - config.risk_low) * risk_score(x)
    uniforms = rng.uniform(size=n)
    risky_outcome = np.where(
        uniforms < probability, config.bad_outcome, config.good_outcome
    )
    outcomes = np.full((n, 2), config.stable_outcome)
    outcomes[np.arange(n), risky_action] = risky_outcome
    mean = np.full((n, 2), config.stable_outcome)
    mean[np.arange(n), risky_action] = (1.0 - probability) * config.good_outcome
    propensity = propensity_from_action(risky_action, config.risky_propensity)
    action = sample_actions(propensity, rng)
    return pack_dataset(
        x, action, outcomes, propensity, mean,
        score=score.reshape(-1, 1).astype(np.float32),
        risky_action=risky_action.reshape(-1, 1),
        catastrophe_probability=probability.reshape(-1, 1).astype(np.float32),
        catastrophe_uniform=uniforms.reshape(-1, 1).astype(np.float32),
    )


def kl_to_uniform(weights):
    weights = np.clip(weights, 1e-300, 1.0)
    return float(np.sum(weights * (np.log(weights) + np.log(weights.size))))


def tilt_weights(score, beta):
    logits = beta * np.asarray(score, dtype=np.float64).reshape(-1)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def calibrated_weights(score, severity):
    target = 0.5 * (SHIFT_SCALE * float(severity)) ** 2
    lower, upper = 0.0, 1.0
    while kl_to_uniform(tilt_weights(score, upper)) < target:
        upper *= 2.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if kl_to_uniform(tilt_weights(score, midpoint)) < target:
            lower = midpoint
        else:
            upper = midpoint
    weights = tilt_weights(score, 0.5 * (lower + upper))
    return weights, kl_to_uniform(weights)


def shift_specification(test, severity, num_environments, replication):
    score = risk_score(test["X"])
    weights, realized_kl = calibrated_weights(score, severity)
    severity_index = SEVERITIES.index(float(severity))
    indices = []
    for environment in range(num_environments):
        seed = 50000 + 10000 * replication + 1000 * severity_index + environment
        indices.append(
            np.random.default_rng(seed).choice(
                test["X"].shape[0], size=test["X"].shape[0], replace=True, p=weights
            )
        )
    return {"indices": np.asarray(indices, dtype=np.int32), "kl": realized_kl}


def drift_basis(x):
    x1, x2 = x[:, 0], x[:, 1]
    basis = np.column_stack(
        [x1, x2, x1 * x2, x1**2 - x2**2, np.sin(np.pi * x1), np.cos(np.pi * x2)]
    )
    basis = (basis - basis.mean(0)) / basis.std(0)
    return np.column_stack([np.ones(x.shape[0]), basis])


def build_susceptibility(x, num_environments, seed=2026):
    basis = drift_basis(np.asarray(x, dtype=np.float64))
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(num_environments, basis.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    return np.exp(0.35 * np.clip(basis @ theta.T, -2.0, 2.0)).T


def bernoulli_kl(probability, baseline):
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    baseline = np.clip(baseline, 1e-12, 1.0 - 1e-12)
    return probability * np.log(probability / baseline) + (
        1.0 - probability
    ) * np.log((1.0 - probability) / (1.0 - baseline))


def solve_drift_probability(susceptibility, target, baseline):
    baseline_logit = np.log(baseline / (1.0 - baseline))

    def probability(tau):
        return sigmoid(baseline_logit + tau * susceptibility)

    lower, upper = 0.0, 1.0
    while bernoulli_kl(probability(upper), baseline).mean() < target:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if bernoulli_kl(probability(midpoint), baseline).mean() < target:
            lower = midpoint
        else:
            upper = midpoint
    return probability(0.5 * (lower + upper))


def drift_specification(test, severity, num_environments):
    baseline = test["catastrophe_probability"].reshape(-1).astype(np.float64)
    susceptibility = build_susceptibility(test["X"], num_environments)
    target = 0.5 * (SHIFT_SCALE * float(severity)) ** 2
    probability = np.stack(
        [solve_drift_probability(row, target, baseline) for row in susceptibility]
    )
    realized = bernoulli_kl(probability, baseline).mean(axis=1)
    return {"probabilities": probability.astype(np.float32), "kl": realized}


def generate_replication(replication, n_train=10000, n_test=2500, num_environments=100):
    data = {"config": asdict(CONFIG), "replication": replication}
    for boundary in ("linear", "nonlinear"):
        data[boundary] = {
            "train": simulate_smooth_dataset(n_train, 42 + replication, boundary),
            "test": simulate_smooth_dataset(n_test, 1042 + replication, boundary),
        }
    data["tail"] = {
        "train": simulate_tail_dataset(n_train, 20042 + replication),
        "test": simulate_tail_dataset(n_test, 21042 + replication),
    }
    test = data["tail"]["test"]
    data["covariate"] = {
        f"{severity:g}": shift_specification(test, severity, num_environments, replication)
        for severity in SEVERITIES
    }
    data["concept"] = {
        f"{severity:g}": drift_specification(test, severity, num_environments)
        for severity in SEVERITIES
    }
    return data


def concept_outcomes(test, probabilities, config=CONFIG):
    n = test["X"].shape[0]
    risky_action = test["risky_action"].reshape(-1)
    uniforms = test["catastrophe_uniform"].reshape(-1)
    risky = np.where(
        uniforms[None, :] < probabilities, config.bad_outcome, config.good_outcome
    )
    outcomes = np.full((probabilities.shape[0], n, 2), config.stable_outcome)
    outcomes[:, np.arange(n), risky_action] = risky
    return outcomes.astype(np.float32)
