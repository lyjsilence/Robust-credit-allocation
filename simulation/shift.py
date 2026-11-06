"""Continuous-tail outcomes and KL-calibrated deployment perturbations."""

import numpy as np
from scipy.special import expit, logit


SEVERITIES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
KL_SCALE = 0.20
P_LOW = 0.005
P_HIGH = 0.780
NOISE_SD = 0.6


def target_kl(severity):
    severity = float(severity)
    return 0.5 * (KL_SCALE * severity) ** 2


def risk_score(x):
    x = np.asarray(x, dtype=np.float64)
    wave = np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])
    return expit(30.0 * (wave - 0.70))


def baseline_probability(x):
    return P_LOW + (P_HIGH - P_LOW) * risk_score(x)


def nominal_outcomes(x, risky_action, uniforms, gaussian_noise):
    """Return continuous potential outcomes and fixed exogenous draws."""
    x = np.asarray(x, dtype=np.float64)
    risky_action = np.asarray(risky_action, dtype=np.int64).reshape(-1)
    uniforms = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    gaussian_noise = np.asarray(gaussian_noise, dtype=np.float64).reshape(-1)
    n = len(x)
    p0 = baseline_probability(x)
    z = expit(gaussian_noise)
    bad = 0.02 * z
    good = 0.88 + 0.04 * z
    stable = 0.77 + 0.04 * z
    risky = np.where(uniforms < p0, bad, good)
    y_all = np.empty((n, 2), dtype=np.float64)
    y_all[np.arange(n), risky_action] = risky
    y_all[np.arange(n), 1 - risky_action] = stable
    mu = np.empty_like(y_all)
    mu[np.arange(n), risky_action] = p0 * 0.01 + (1 - p0) * 0.90
    mu[np.arange(n), 1 - risky_action] = 0.79
    return {
        "Y_all": y_all,
        "mu": mu,
        "p0": p0,
        "U": uniforms,
        "Z": z,
        "risky_action": risky_action,
    }


def bernoulli_kl(p, p0):
    p = np.asarray(p, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    return p * np.log(p / p0) + (1 - p) * np.log((1 - p) / (1 - p0))


def drift_basis(x):
    x = np.asarray(x, dtype=np.float64)
    x1, x2 = x[:, 0], x[:, 1]
    nonconstant = np.column_stack(
        (x1, x2, x1 * x2, x1**2 - x2**2, np.sin(np.pi * x1), np.cos(np.pi * x2))
    )
    scales = nonconstant.std(axis=0)
    if np.any(scales <= 0):
        raise ValueError("Zero-scale drift feature")
    nonconstant = (nonconstant - nonconstant.mean(axis=0)) / scales
    return np.column_stack((np.ones(len(x)), nonconstant))


def susceptibility(x, direction_seeds):
    basis = drift_basis(x)
    rows = []
    for seed in direction_seeds:
        theta = np.random.default_rng(int(seed)).normal(size=basis.shape[1])
        theta /= np.linalg.norm(theta)
        rows.append(np.exp(0.35 * np.clip(basis @ theta, -2, 2)))
    return np.asarray(rows, dtype=np.float64)


def solve_concept_probability(p0, h, severity):
    """Calibrate each environment's average Bernoulli KL exactly."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    target = target_kl(severity)
    if target == 0:
        return p0.copy(), 0.0, 0.0
    baseline_logit = logit(p0)
    prob = lambda tau: expit(baseline_logit + tau * h)
    lo, hi = 0.0, 1.0
    while np.mean(bernoulli_kl(prob(hi), p0)) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Cannot bracket concept KL root")
    for _ in range(70):
        mid = (lo + hi) / 2
        if np.mean(bernoulli_kl(prob(mid), p0)) < target:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    p = prob(tau)
    realized = float(np.mean(bernoulli_kl(p, p0)))
    if abs(realized - target) > 1e-9:
        raise AssertionError("Concept KL calibration failed")
    return p, float(tau), realized


def tilt_weights(x, severity):
    """Calibrate finite-sample covariate KL to the uniform test measure."""
    r = risk_score(x)
    target = target_kl(severity)
    n = len(r)
    if target == 0:
        return np.full(n, 1.0 / n), 0.0, 0.0

    def weights(beta):
        scores = beta * r
        unnormalized = np.exp(scores - scores.max())
        return unnormalized / unnormalized.sum()

    def kl(q):
        return float(np.sum(q * np.log(n * q)))

    lo, hi = 0.0, 1.0
    while kl(weights(hi)) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Cannot bracket covariate KL root")
    for _ in range(70):
        mid = (lo + hi) / 2
        if kl(weights(mid)) < target:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    q = weights(beta)
    realized = kl(q)
    if abs(realized - target) > 1e-9:
        raise AssertionError("Covariate KL calibration failed")
    return q, float(beta), realized


def concept_outcomes(risky_action, uniforms, z, probabilities):
    """Potential outcomes for one fixed-X deployment environment."""
    risky_action = np.asarray(risky_action, dtype=np.int64).reshape(-1)
    uniforms = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    n = len(risky_action)
    risky = np.where(uniforms < probabilities, 0.02 * z, 0.88 + 0.04 * z)
    stable = 0.77 + 0.04 * z
    outcomes = np.empty((n, 2), dtype=np.float64)
    outcomes[np.arange(n), risky_action] = risky
    outcomes[np.arange(n), 1 - risky_action] = stable
    return outcomes
