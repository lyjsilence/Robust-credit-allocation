"""Rare-catastrophe DGP with exact Bernoulli-KL concept drift."""

from __future__ import annotations

import numpy as np

from generate_dgp import (
    exponential_tilt_weights,
    nonlinear_score,
    sample_actions,
    sample_uniform_ball,
)


P0 = 0.10
Y_BAD = 0.00
Y_GOOD = 0.90
Y_STABLE = 0.79
P_LOG_RISKY = 0.75


def simulate_catastrophe_dataset(
    n: int,
    seed: int,
    *,
    retain_drift_state: bool = False,
) -> dict[str, np.ndarray]:
    """Generate logged data with a high-mean, lower-tail-risky action."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    score = nonlinear_score(x[:, 0], x[:, 1])
    risky_action = (score >= 0.0).astype(np.int64)

    catastrophe_uniform = rng.uniform(size=n)
    risky_outcome = np.where(
        catastrophe_uniform < P0, Y_BAD, Y_GOOD
    )
    outcomes = np.full((n, 2), Y_STABLE, dtype=np.float64)
    outcomes[np.arange(n), risky_action] = risky_outcome

    expected_outcomes = np.full((n, 2), Y_STABLE, dtype=np.float64)
    expected_risky = P0 * Y_BAD + (1.0 - P0) * Y_GOOD
    expected_outcomes[np.arange(n), risky_action] = expected_risky

    propensity = np.full((n, 2), 1.0 - P_LOG_RISKY, dtype=np.float64)
    propensity[np.arange(n), risky_action] = P_LOG_RISKY
    action = sample_actions(propensity, rng)
    observed_outcome = outcomes[np.arange(n), action]
    observed_propensity = propensity[np.arange(n), action]

    data = {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1),
        "Y": observed_outcome.reshape(-1, 1).astype(np.float32),
        "prop_score": observed_propensity.reshape(-1, 1).astype(np.float32),
        "Y_all": outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": expected_outcomes.astype(np.float32),
    }
    if retain_drift_state:
        data["score"] = score.reshape(-1, 1).astype(np.float32)
        data["risky_action"] = risky_action.reshape(-1, 1)
        data["catastrophe_uniform"] = catastrophe_uniform.reshape(-1, 1)
    return data


def drift_basis(x: np.ndarray) -> np.ndarray:
    """Return the standardized basis used for heterogeneous susceptibility."""
    x1, x2 = x[:, 0], x[:, 1]
    continuous = np.column_stack(
        [
            x1,
            x2,
            x1 * x2,
            x1**2 - x2**2,
            np.sin(np.pi * x1),
            np.cos(np.pi * x2),
        ]
    )
    continuous = (continuous - continuous.mean(0)) / continuous.std(0)
    return np.column_stack([np.ones(x.shape[0]), continuous])


def build_susceptibility(
    x: np.ndarray,
    n_environments: int,
    *,
    seed: int = 2026,
    heterogeneity_scale: float = 0.35,
) -> np.ndarray:
    """Build positive susceptibility for each deployment environment."""
    basis = drift_basis(np.asarray(x, dtype=np.float64))
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_environments, basis.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    raw = basis @ theta.T
    return np.exp(
        heterogeneity_scale * np.clip(raw, -2.0, 2.0)
    ).T


def bernoulli_kl(probability: np.ndarray, baseline: float = P0) -> np.ndarray:
    """Compute KL(Bernoulli(probability) || Bernoulli(baseline))."""
    probability = np.clip(
        np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12
    )
    return (
        probability * np.log(probability / baseline)
        + (1.0 - probability)
        * np.log((1.0 - probability) / (1.0 - baseline))
    )


def solve_catastrophe_probability(
    susceptibility: np.ndarray,
    eta_test: float,
    *,
    baseline: float = P0,
) -> np.ndarray:
    """Solve the conditional catastrophe probabilities for one environment."""
    susceptibility = np.asarray(susceptibility, dtype=np.float64)
    if eta_test == 0.0:
        return np.full_like(susceptibility, baseline)

    baseline_logit = np.log(baseline / (1.0 - baseline))

    def probabilities(tau: float) -> np.ndarray:
        logits = baseline_logit + tau * susceptibility
        return 1.0 / (1.0 + np.exp(-logits))

    lower, upper = 0.0, 1.0
    while bernoulli_kl(probabilities(upper), baseline).mean() < eta_test:
        upper *= 2.0

    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        realized_kl = bernoulli_kl(
            probabilities(midpoint), baseline
        ).mean()
        if realized_kl < eta_test:
            lower = midpoint
        else:
            upper = midpoint
    return probabilities(0.5 * (lower + upper))


def environment_probabilities(
    susceptibility: np.ndarray,
    lambda_2: float,
    *,
    shift_scale: float = 0.20,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return calibrated probabilities and realized KL by environment."""
    eta_test = 0.5 * (shift_scale * float(lambda_2)) ** 2
    probabilities = np.stack(
        [
            solve_catastrophe_probability(row, eta_test)
            for row in susceptibility
        ]
    )
    realized_kl = bernoulli_kl(probabilities).mean(axis=1)
    return probabilities, eta_test, realized_kl


def drifted_outcomes(
    catastrophe_uniform: np.ndarray,
    risky_action: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Generate potential outcomes using common catastrophe uniforms."""
    catastrophe_uniform = np.asarray(catastrophe_uniform).reshape(-1)
    risky_action = np.asarray(risky_action).reshape(-1)
    catastrophe = catastrophe_uniform[None, :] < probabilities
    risky_outcome = np.where(catastrophe, Y_BAD, Y_GOOD)
    risky_indicator = np.eye(2)[risky_action]
    return (
        Y_STABLE
        + risky_indicator[None, :, :]
        * (risky_outcome[:, :, None] - Y_STABLE)
    )


def covariate_environment_values(
    outcomes: np.ndarray,
    policy_actions: np.ndarray,
    severity: float,
    n_environments: int,
    replication: int,
    *,
    base_test_seed: int = 1042,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fixed policies under exponential-tilting resampling."""
    outcomes = np.asarray(outcomes)
    policy_actions = np.atleast_2d(policy_actions)
    rows = np.arange(outcomes.shape[0])
    selected = np.stack(
        [outcomes[rows, actions] for actions in policy_actions]
    )

    if float(severity) == 0.0:
        values = selected.mean(axis=1, keepdims=True)
        return np.repeat(values, n_environments, axis=1), np.zeros(
            n_environments
        )

    values = np.empty((policy_actions.shape[0], n_environments))
    divergences = np.empty(n_environments)
    for environment in range(n_environments):
        seed = base_test_seed + 10000 * (replication + 1) + environment
        rng = np.random.default_rng(seed)
        weights, divergences[environment] = exponential_tilt_weights(
            rng, outcomes.shape[0], severity
        )
        indices = rng.choice(
            outcomes.shape[0],
            size=outcomes.shape[0],
            replace=True,
            p=weights,
        )
        values[:, environment] = selected[:, indices].mean(axis=1)
    return values, divergences


def population_worst_risky_value(eta: float = 0.20) -> float:
    """Return the two-point risky action's KL worst-case value."""
    probability = solve_catastrophe_probability(np.ones(1), eta)[0]
    return float(probability * Y_BAD + (1.0 - probability) * Y_GOOD)
