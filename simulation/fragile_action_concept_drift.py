"""Fragile high-return action DGP with KL-calibrated concept drift."""

from __future__ import annotations

import numpy as np

from generate_dgp import (
    nonlinear_score,
    sample_actions,
    sample_uniform_ball,
    sigmoid,
)
from latent_gaussian_concept_drift import drift_basis


def simulate_fragile_dataset(
    n: int,
    seed: int,
    *,
    mu_stable: float = 0.60,
    sigma_stable: float = 0.20,
    sigma_fragile: float = 0.80,
    p_fragile: float = 0.75,
    retain_drift_state: bool = False,
) -> dict[str, np.ndarray]:
    """Generate logged data with a mean-versus-downside-risk trade-off."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    score = nonlinear_score(x[:, 0], x[:, 1])
    contrast = np.tanh(6.0 * score)
    fragile_action = (contrast >= 0.0).astype(np.int64)

    mu = np.full((n, 2), mu_stable, dtype=np.float64)
    mu_fragile = 0.80 + 0.20 * np.abs(contrast)
    mu[np.arange(n), fragile_action] = mu_fragile

    sigma = np.full((n, 2), sigma_stable, dtype=np.float64)
    sigma[np.arange(n), fragile_action] = sigma_fragile
    noise = rng.normal(size=(n, 2)) * sigma
    potential_outcomes = sigmoid(mu + noise)

    propensity = np.full((n, 2), 1.0 - p_fragile, dtype=np.float64)
    propensity[np.arange(n), fragile_action] = p_fragile
    action = sample_actions(propensity, rng)
    observed_outcome = potential_outcomes[np.arange(n), action]
    observed_propensity = propensity[np.arange(n), action]

    data = {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1),
        "Y": observed_outcome.reshape(-1, 1).astype(np.float32),
        "prop_score": observed_propensity.reshape(-1, 1).astype(np.float32),
        "Y_all": potential_outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": mu.astype(np.float32),
    }
    if retain_drift_state:
        data["eps_all"] = noise.astype(np.float32)
        data["sigma_all"] = sigma.astype(np.float32)
        data["score"] = score.reshape(-1, 1).astype(np.float32)
        data["fragile_action"] = fragile_action.reshape(-1, 1)
    return data


def build_fragile_directions(
    x: np.ndarray,
    n_environments: int,
    *,
    seed: int = 2026,
    heterogeneity_scale: float = 0.35,
) -> np.ndarray:
    """Return positive directions with mean(v_m(X)^2) = 1."""
    basis = drift_basis(np.asarray(x, dtype=np.float64))
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_environments, basis.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    raw = basis @ theta.T
    direction = np.exp(
        heterogeneity_scale * np.clip(raw, -2.0, 2.0)
    )
    direction /= np.sqrt(np.mean(direction**2, axis=0))
    return direction.T


def drifted_fragile_outcomes(
    mu: np.ndarray,
    errors: np.ndarray,
    sigma: np.ndarray,
    fragile_action: np.ndarray,
    directions: np.ndarray,
    lambda_2: float,
    *,
    shift_scale: float = 0.20,
) -> tuple[np.ndarray, float]:
    """Lower only the fragile action's latent mean."""
    kappa = shift_scale * float(lambda_2)
    fragile_action = np.asarray(fragile_action).reshape(-1)
    fragile_indicator = np.eye(2)[fragile_action]
    mean_shift = (
        -kappa
        * directions[:, :, None]
        * np.asarray(sigma, dtype=np.float64)[None, :, :]
        * fragile_indicator[None, :, :]
    )
    latent = (
        np.asarray(mu, dtype=np.float64)[None, :, :]
        + np.asarray(errors, dtype=np.float64)[None, :, :]
        + mean_shift
    )
    return sigmoid(latent), 0.5 * kappa**2


def empirical_latent_kl(
    directions: np.ndarray, lambda_2: float, shift_scale: float = 0.20
) -> np.ndarray:
    """Return the latent KL radius in each environment."""
    kappa = shift_scale * float(lambda_2)
    return 0.5 * kappa**2 * np.mean(directions**2, axis=1)
