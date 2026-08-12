"""Generate the simulation data used in the paper.

The script exposes the three simulation modes directly:

- ``raw``: no deployment shift;
- ``shift``: covariate shift through an X-dependent exponential tilt;
- ``drift``: concept drift through the lower-tail-risk outcome model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np


DGP_MODES = ("raw", "shift", "drift")
MODE_ALIASES = {
    "raw": "raw",
    "none": "raw",
    "shift": "shift",
    "covariate": "shift",
    "drift": "drift",
    "concept": "drift",
}
SHIFT_SCALE = 0.20

P0 = 0.10
Y_BAD = 0.00
Y_GOOD = 0.90
Y_STABLE = 0.79
P_LOG_RISKY = 0.75


def sample_uniform_ball(
    n: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample uniformly from the closed unit Euclidean ball in R^d."""
    z = rng.normal(size=(n, d))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    directions = z / np.maximum(norms, 1e-12)
    radii = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return directions * radii


def sample_actions(
    probabilities: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample one categorical action from each row of a probability matrix."""
    n, num_actions = probabilities.shape
    actions = np.empty(n, dtype=np.int64)
    for i in range(n):
        actions[i] = rng.choice(num_actions, p=probabilities[i])
    return actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def linear_score(
    x1: np.ndarray,
    x2: np.ndarray,
    w1: float,
    w2: float,
    intercept: float,
) -> np.ndarray:
    """Return the score whose zero contour is the linear decision boundary."""
    return intercept + w1 * x1 + w2 * x2


def nonlinear_boundary(x2: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Return the S-shaped boundary constructed from two semicircles."""
    half_radius = 0.5 * radius
    x2 = np.asarray(x2)
    boundary = np.empty_like(x2, dtype=np.float64)

    upper = x2 >= 0.0
    upper_term = half_radius**2 - (x2[upper] - half_radius) ** 2
    boundary[upper] = np.sqrt(np.maximum(upper_term, 0.0))

    lower = ~upper
    lower_term = half_radius**2 - (x2[lower] + half_radius) ** 2
    boundary[lower] = -np.sqrt(np.maximum(lower_term, 0.0))
    return boundary


def nonlinear_score(
    x1: np.ndarray, x2: np.ndarray, radius: float = 1.0
) -> np.ndarray:
    """Return the score whose zero contour is the nonlinear decision boundary."""
    return x1 - nonlinear_boundary(x2, radius=radius)


def latent_means(
    score: np.ndarray, baseline: float, effect: float, sharpness: float
) -> np.ndarray:
    """Construct latent conditional means for the two actions."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    contrast = effect * np.tanh(sharpness * score)
    return np.stack([baseline - contrast, baseline + contrast], axis=1)


def simulate_dataset(
    n: int,
    seed: int,
    g_type: str,
    *,
    sigmas: tuple[float, float] = (0.3, 0.3),
    p_hi: float = 0.75,
    w1: float = 1.0,
    w2: float = -0.8,
    b: float = 0.0,
    R: float = 1.0,
    gamma: float = 6.0,
    m0: float = 0.0,
    delta: float = 1.5,
    retain_drift_state: bool = False,
) -> dict[str, np.ndarray]:
    """Generate one logged dataset under the nominal environment."""
    rng = np.random.default_rng(seed)
    num_actions = 2

    x = sample_uniform_ball(n, 5, rng).astype(np.float64)
    x1, x2 = x[:, 0], x[:, 1]
    if g_type == "linear":
        score = linear_score(x1, x2, w1=w1, w2=w2, intercept=b)
    elif g_type == "nonlinear":
        score = nonlinear_score(x1, x2, radius=R)
    else:
        raise ValueError("g_type must be 'linear' or 'nonlinear'")

    mu = latent_means(score, baseline=m0, effect=delta, sharpness=gamma)
    sigma_array = np.asarray(sigmas, dtype=np.float64).reshape(1, num_actions)
    noise = rng.normal(size=(n, num_actions)).astype(np.float64) * sigma_array
    potential_outcomes = sigmoid(mu + noise)

    optimal_action = np.argmax(mu, axis=1)
    propensity = np.empty((n, num_actions), dtype=np.float64)
    propensity[optimal_action == 0] = np.array([p_hi, 1.0 - p_hi])
    propensity[optimal_action == 1] = np.array([1.0 - p_hi, p_hi])

    action = sample_actions(propensity, rng)
    observed_outcome = potential_outcomes[np.arange(n), action]
    observed_propensity = propensity[np.arange(n), action]

    dataset = {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1).astype(np.int64),
        "Y": observed_outcome.reshape(-1, 1).astype(np.float32),
        "prop_score": observed_propensity.reshape(-1, 1).astype(np.float32),
        "Y_all": potential_outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": mu.astype(np.float32),
    }
    if retain_drift_state:
        dataset["eps_all"] = noise.astype(np.float32)
        dataset["score"] = score.reshape(-1, 1).astype(np.float32)
    return dataset


def kl_to_uniform(weights: np.ndarray) -> float:
    """Compute KL(weights || uniform)."""
    n = weights.size
    clipped = np.clip(weights, 1e-300, 1.0)
    return float(np.sum(clipped * (np.log(clipped) + np.log(n))))


def middle_margin_score(score: np.ndarray, gamma: float) -> np.ndarray:
    """Emphasize observations with an intermediate decision margin."""
    margin = np.abs(np.asarray(score, dtype=np.float64))
    return margin * np.exp(-float(gamma) * margin)


def tilt_weights(score: np.ndarray, beta: float) -> np.ndarray:
    """Return exponential-tilting weights for a fixed shift score."""
    logits = float(beta) * np.asarray(score, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights


def calibrated_tilt_weights(
    score: np.ndarray, severity: float
) -> tuple[np.ndarray, float]:
    """Calibrate structured tilt weights to the requested KL level."""
    target_kl = 0.5 * (SHIFT_SCALE * float(severity)) ** 2
    if target_kl == 0.0:
        n = np.asarray(score).size
        return np.full(n, 1.0 / n, dtype=np.float64), 0.0

    lower, upper = 0.0, 1.0
    while kl_to_uniform(tilt_weights(score, upper)) < target_kl:
        upper *= 2.0

    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        weights = tilt_weights(score, midpoint)
        if kl_to_uniform(weights) < target_kl:
            lower = midpoint
        else:
            upper = midpoint

    weights = tilt_weights(score, 0.5 * (lower + upper))
    return weights, kl_to_uniform(weights)


def apply_covariate_shift(
    base_data: dict[str, np.ndarray],
    weights: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample a test dataset using structured exponential-tilting weights."""
    n = base_data["X"].shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=n, replace=True, p=weights)
    return {
        key: value[indices].copy()
        if isinstance(value, np.ndarray) and value.shape[0] == n
        else value
        for key, value in base_data.items()
    }


# Lower-tail-risk DGP used by the revised Table 2.
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
    shift_scale: float = SHIFT_SCALE,
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


def generate_raw_replication(
    replication: int,
    *,
    n_train: int,
    n_test: int,
    g_type: str,
    base_seed_train: int,
    base_seed_test: int,
    dgp_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Generate the unshifted smooth-outcome training and test datasets."""
    train = simulate_dataset(
        n_train,
        base_seed_train + replication,
        g_type,
        **dgp_kwargs,
    )
    test_base = simulate_dataset(
        n_test,
        base_seed_test + replication,
        g_type,
        **dgp_kwargs,
    )
    return {"train": train, "test": test_base}


def generate_shift_replication(
    replication: int,
    *,
    n_train: int,
    n_test: int,
    g_type: str,
    severity: float,
    num_environments: int,
    base_seed_train: int,
    base_seed_test: int,
    dgp_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Generate smooth outcomes under X-dependent covariate shift."""
    data = generate_raw_replication(
        replication,
        n_train=n_train,
        n_test=n_test,
        g_type=g_type,
        base_seed_train=base_seed_train,
        base_seed_test=base_seed_test,
        dgp_kwargs=dgp_kwargs,
    )
    train, test_base = data["train"], data["test"]

    x = test_base["X"]
    if g_type == "linear":
        raw_score = linear_score(
            x[:, 0],
            x[:, 1],
            w1=dgp_kwargs.get("w1", 1.0),
            w2=dgp_kwargs.get("w2", -0.8),
            intercept=dgp_kwargs.get("b", 0.0),
        )
    else:
        raw_score = nonlinear_score(
            x[:, 0], x[:, 1], radius=dgp_kwargs.get("R", 1.0)
        )
    shift_score = middle_margin_score(
        raw_score, gamma=dgp_kwargs.get("gamma", 6.0)
    )
    q, realized_kl = calibrated_tilt_weights(shift_score, severity)
    environment_seeds = (
        base_seed_test + 10000 * (replication + 1) + j
        for j in range(num_environments)
    )
    environments, weights, divergences = [], [], []
    for seed in environment_seeds:
        environments.append(apply_covariate_shift(test_base, q, seed))
        weights.append(q)
        divergences.append(realized_kl)
    test = {
        "base": test_base,
        "envs": environments,
        "q_list": weights,
        "kl_list": divergences,
        "shift_eta": float(severity),
    }
    return {"train": train, "test": test}


def generate_drift_replication(
    replication: int,
    *,
    n_train: int,
    n_test: int,
    severity: float,
    num_environments: int,
    base_seed_train: int,
    base_seed_test: int,
) -> dict[str, Any]:
    """Generate lower-tail outcomes under concept drift."""
    train = simulate_catastrophe_dataset(
        n_train, base_seed_train + replication
    )
    test_base = simulate_catastrophe_dataset(
        n_test,
        base_seed_test + replication,
        retain_drift_state=True,
    )
    susceptibility = build_susceptibility(test_base["X"], num_environments)
    probabilities, eta_test, divergences = environment_probabilities(
        susceptibility, severity
    )
    outcomes = drifted_outcomes(
        test_base["catastrophe_uniform"],
        test_base["risky_action"],
        probabilities,
    )

    rows = np.arange(n_test)
    actions = test_base["T"].reshape(-1)
    risky_actions = test_base["risky_action"].reshape(-1)
    environments = []
    for probability, outcome in zip(probabilities, outcomes):
        environment = dict(test_base)
        environment["Y_all"] = outcome
        environment["Y"] = (
            outcome[rows, actions].reshape(-1, 1).astype(np.float32)
        )
        expected = np.full((n_test, 2), Y_STABLE)
        expected[rows, risky_actions] = (
            probability * Y_BAD + (1.0 - probability) * Y_GOOD
        )
        environment["mu"] = expected.astype(np.float32)
        environments.append(environment)

    return {
        "train": train,
        "test": {
            "base": test_base,
            "envs": environments,
            "lambda2": float(severity),
            "eta_test": eta_test,
            "kl_list": divergences,
        },
    }


def normalize_mode(mode: str) -> str:
    """Return the canonical raw/shift/drift mode name."""
    try:
        return MODE_ALIASES[mode]
    except KeyError as error:
        choices = ", ".join(DGP_MODES)
        raise ValueError(f"mode must be one of: {choices}") from error


def generate_replication(
    replication: int,
    *,
    n_train: int,
    n_test: int,
    g_type: str,
    mode: str,
    severity: float,
    num_environments: int,
    base_seed_train: int,
    base_seed_test: int,
    dgp_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch one replication explicitly to raw, shift, or drift."""
    mode = normalize_mode(mode)
    if mode == "raw":
        return generate_raw_replication(
            replication,
            n_train=n_train,
            n_test=n_test,
            g_type=g_type,
            base_seed_train=base_seed_train,
            base_seed_test=base_seed_test,
            dgp_kwargs=dgp_kwargs,
        )
    if mode == "shift":
        return generate_shift_replication(
            replication,
            n_train=n_train,
            n_test=n_test,
            g_type=g_type,
            severity=severity,
            num_environments=num_environments,
            base_seed_train=base_seed_train,
            base_seed_test=base_seed_test,
            dgp_kwargs=dgp_kwargs,
        )
    return generate_drift_replication(
        replication,
        n_train=n_train,
        n_test=n_test,
        severity=severity,
        num_environments=num_environments,
        base_seed_train=base_seed_train,
        base_seed_test=base_seed_test,
    )


def plot_decision_regions(
    g_type: str,
    output_path: Path,
    *,
    seed: int,
    n_plot: int,
    w1: float,
    w2: float,
    b: float,
    R: float,
    gamma: float,
    m0: float,
    delta: float,
) -> None:
    """Plot the Bayes-optimal decision regions in the first two covariates."""
    import matplotlib.pyplot as plt

    import basics_with_holdout as utils

    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n_plot, 5, rng)
    if g_type == "linear":
        score = linear_score(x[:, 0], x[:, 1], w1=w1, w2=w2, intercept=b)
    else:
        score = nonlinear_score(x[:, 0], x[:, 1], radius=R)
    decision = np.argmax(
        latent_means(score, baseline=m0, effect=delta, sharpness=gamma), axis=1
    )

    utils.set_large_fonts()
    plt.figure(figsize=(6.5, 6))
    plt.scatter(x[:, 0], x[:, 1], c=decision, s=6, alpha=0.85)
    plt.xlim([-1.05, 1.05])
    plt.ylim([-1.05, 1.05])
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate nominal, covariate-shift, or concept-drift data."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=DGP_MODES,
        help="DGP mode: raw (none), shift (covariates), or drift (concept).",
    )
    mode_group.add_argument(
        "--shift_type",
        choices=("none", "covariate", "concept"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument(
        "--g_type", choices=("linear", "nonlinear"), default="nonlinear"
    )
    parser.add_argument("--num_rep", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=2500)
    parser.add_argument("--num_environments", type=int, default=100)
    parser.add_argument("--out_dir", type=Path, default=Path("./data"))
    parser.add_argument("--base_seed_train", type=int, default=42)
    parser.add_argument("--base_seed_test", type=int, default=1042)
    parser.add_argument("--p_hi", type=float, default=0.75)
    parser.add_argument("--sigma0", type=float, default=0.3)
    parser.add_argument("--sigma1", type=float, default=0.3)
    parser.add_argument("--m0", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=6.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=-0.8)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--R", type=float, default=1.0)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no_plot", dest="plot", action="store_false")
    parser.set_defaults(plot=False)
    parser.add_argument("--plot_n", type=int, default=6000)
    return parser


def output_stem(mode: str, severity: float) -> str:
    mode = normalize_mode(mode)
    if mode == "raw":
        return "Shift_0.0"
    prefix = "Shift" if mode == "shift" else "Drift"
    severity = float(severity)
    label = str(int(severity)) if severity.is_integer() else str(severity)
    return f"{prefix}_{label}"


def output_directory(
    base_dir: Path, g_type: str, mode: str, severity: float
) -> Path:
    """Match the condition directories read by run_experiment.py."""
    mode = normalize_mode(mode)
    if mode == "raw":
        condition = g_type
    else:
        condition = f"{g_type}_{mode}={float(severity):.1f}"
    return base_dir / condition


def main() -> None:
    args = build_parser().parse_args()
    mode = normalize_mode(args.mode or args.shift_type or "drift")

    output_dir = output_directory(
        args.out_dir, args.g_type, mode, args.severity
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dgp_kwargs = {
        "sigmas": (args.sigma0, args.sigma1),
        "p_hi": args.p_hi,
        "w1": args.w1,
        "w2": args.w2,
        "b": args.b,
        "R": args.R,
        "gamma": args.gamma,
        "m0": args.m0,
        "delta": args.delta,
    }
    n_test = 10000 if mode == "raw" else args.n_test
    stem = output_stem(mode, args.severity)

    for replication in range(args.num_rep):
        data = generate_replication(
            replication,
            n_train=args.n_train,
            n_test=n_test,
            g_type=args.g_type,
            mode=mode,
            severity=args.severity,
            num_environments=args.num_environments,
            base_seed_train=args.base_seed_train,
            base_seed_test=args.base_seed_test,
            dgp_kwargs=dgp_kwargs,
        )
        output_path = output_dir / f"{stem}-exp-{replication}.pkl"
        joblib.dump(data, output_path)
        print(f"Saved {output_path}")

    if args.plot:
        plot_decision_regions(
            args.g_type,
            output_dir / f"decision_regions_{args.g_type}.png",
            seed=123,
            n_plot=args.plot_n,
            w1=args.w1,
            w2=args.w2,
            b=args.b,
            R=args.R,
            gamma=args.gamma,
            m0=args.m0,
            delta=args.delta,
        )


if __name__ == "__main__":
    main()
