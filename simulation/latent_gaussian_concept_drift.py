"""KL-calibrated latent-Gaussian concept drift for the Table 2 experiment."""

from __future__ import annotations

import numpy as np

from generate_dgp import sigmoid


def drift_basis(x: np.ndarray) -> np.ndarray:
    """Return a small standardized basis for heterogeneous drift directions."""
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
    continuous = (continuous - continuous.mean(axis=0)) / continuous.std(axis=0)
    return np.column_stack([np.ones(x.shape[0]), continuous])


def build_drift_directions(
    x: np.ndarray,
    u_base: np.ndarray,
    n_environments: int,
    *,
    seed: int = 2026,
    heterogeneity_scale: float = 0.35,
) -> np.ndarray:
    """Build normalized action-heterogeneous directions.

    The returned array has shape ``(environment, observation, action)`` and
    satisfies mean(v_0^2 + v_1^2) = 1 in every environment.
    """
    basis = drift_basis(np.asarray(x, dtype=np.float64))
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_environments, basis.shape[1]))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)

    raw = basis @ theta.T
    multiplier = np.exp(heterogeneity_scale * np.clip(raw, -2.0, 2.0))
    direction = np.asarray(u_base, dtype=np.float64).reshape(-1, 1) * multiplier
    normalizer = np.sqrt(2.0 * np.mean(direction**2, axis=0))

    v0 = direction / normalizer
    v1 = -v0
    return np.stack([v0.T, v1.T], axis=2)


def drifted_outcomes(
    mu: np.ndarray,
    errors: np.ndarray,
    directions: np.ndarray,
    lambda_2: float,
    *,
    shift_scale: float = 0.20,
    sigmas: tuple[float, float] = (0.3, 0.3),
) -> tuple[np.ndarray, float]:
    """Apply a calibrated latent mean shift and return potential outcomes.

    With unchanged Gaussian covariance, the empirical average conditional KL
    is ``eta_test = 0.5 * (shift_scale * lambda_2) ** 2``.
    """
    kappa = shift_scale * float(lambda_2)
    sigma = np.asarray(sigmas, dtype=np.float64).reshape(1, 1, 2)
    latent = (
        np.asarray(mu, dtype=np.float64)[None, :, :]
        + np.asarray(errors, dtype=np.float64)[None, :, :]
        + kappa * sigma * directions
    )
    return sigmoid(latent), 0.5 * kappa**2


def empirical_latent_kl(
    directions: np.ndarray, lambda_2: float, shift_scale: float = 0.20
) -> np.ndarray:
    """Return the empirical latent KL radius in each drift environment."""
    kappa = shift_scale * float(lambda_2)
    squared_norm = np.mean(np.sum(directions**2, axis=2), axis=1)
    return 0.5 * kappa**2 * squared_norm
