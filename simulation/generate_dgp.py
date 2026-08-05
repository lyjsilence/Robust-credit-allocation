"""Generate the simulation data used in the paper.

The script supports the nominal environment, covariate shift, and concept
drift through one command-line interface. Defaults reproduce the data-generating
process reported in the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np


SHIFT_TYPES = ("none", "covariate", "concept")


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


def exponential_tilt_weights(
    rng: np.random.Generator,
    n: int,
    severity: float,
    center_z: bool = False,
) -> tuple[np.ndarray, float]:
    """Generate exponential-tilting weights and their realized KL divergence."""
    severity = float(severity)
    if severity == 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64), 0.0

    z = rng.normal(size=n).astype(np.float64)
    if center_z:
        z = z - z.mean()
    logits = severity * z
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, kl_to_uniform(weights)


def apply_covariate_shift(
    base_data: dict[str, np.ndarray],
    severity: float,
    seed: int,
    center_z: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Resample a test dataset using exponential-tilting weights."""
    n = base_data["X"].shape[0]
    rng = np.random.default_rng(seed)
    weights, realized_kl = exponential_tilt_weights(
        rng, n=n, severity=severity, center_z=center_z
    )
    if float(severity) == 0.0:
        return base_data, weights, realized_kl

    indices = rng.choice(n, size=n, replace=True, p=weights)
    shifted = {
        key: value[indices].copy()
        if isinstance(value, np.ndarray) and value.shape[0] == n
        else value
        for key, value in base_data.items()
    }
    return shifted, weights, realized_kl


def rotate_vector(w1: float, w2: float, angle: float) -> tuple[float, float]:
    """Rotate a two-dimensional vector."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        float(cosine * w1 - sine * w2),
        float(sine * w1 + cosine * w2),
    )


def apply_concept_drift(
    base_data: dict[str, np.ndarray],
    severity: float,
    seed: int,
    *,
    gamma: float,
    m0: float,
    delta: float,
    g_type: str,
    w1: float,
    w2: float,
    b: float,
    R: float,
    max_severity: float = 5.0,
    max_rotation: float = 1.0,
    max_intercept_shift: float = 0.5,
    max_radius_shift: float = 0.8,
    max_x2_shift: float = 0.3,
    min_radius: float = 0.7,
) -> dict[str, np.ndarray]:
    """Perturb the conditional outcome model while holding covariates fixed."""
    if float(severity) == 0.0:
        return base_data

    rng = np.random.default_rng(seed)
    x = base_data["X"].astype(np.float64)
    action = base_data["T"].reshape(-1).astype(np.int64)
    propensity = base_data["prop_score_all"].astype(np.float64)
    noise = base_data["eps_all"].astype(np.float64)
    x1, x2 = x[:, 0], x[:, 1]

    scaled_severity = float(np.clip(float(severity) / max_severity, 0.0, 1.0))
    rotation_sign = rng.choice([-1.0, 1.0])
    intercept_sign = rng.choice([-1.0, 1.0])
    radius_sign = rng.choice([-1.0, 1.0])
    x2_sign = rng.choice([-1.0, 1.0])

    if g_type == "linear":
        angle = rotation_sign * max_rotation * scaled_severity
        drift_w1, drift_w2 = rotate_vector(w1, w2, angle)
        drift_b = float(b + intercept_sign * max_intercept_shift * scaled_severity)
        score = linear_score(
            x1, x2, w1=drift_w1, w2=drift_w2, intercept=drift_b
        )
    elif g_type == "nonlinear":
        drift_radius = float(
            max(min_radius, float(R) + radius_sign * max_radius_shift * scaled_severity)
        )
        x2_shift = float(x2_sign * max_x2_shift * scaled_severity)
        drift_b = float(b + intercept_sign * max_intercept_shift * scaled_severity)
        score = x1 - nonlinear_boundary(x2 - x2_shift, radius=drift_radius) - drift_b
    else:
        raise ValueError("g_type must be 'linear' or 'nonlinear'")

    mu = latent_means(score, baseline=m0, effect=delta, sharpness=gamma)
    potential_outcomes = sigmoid(mu + noise)
    observed_outcome = potential_outcomes[np.arange(x.shape[0]), action]
    observed_propensity = propensity[np.arange(x.shape[0]), action]

    drifted = dict(base_data)
    drifted["Y_all"] = potential_outcomes.astype(np.float32)
    drifted["Y"] = observed_outcome.reshape(-1, 1).astype(np.float32)
    drifted["prop_score"] = observed_propensity.reshape(-1, 1).astype(np.float32)
    drifted["mu"] = mu.astype(np.float32)
    return drifted


def generate_replication(
    replication: int,
    *,
    n_train: int,
    n_test: int,
    g_type: str,
    shift_type: str,
    severity: float,
    num_environments: int,
    center_z: bool,
    base_seed_train: int,
    base_seed_test: int,
    dgp_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Generate one training dataset and its corresponding test environment(s)."""
    if shift_type not in SHIFT_TYPES:
        raise ValueError(f"shift_type must be one of {SHIFT_TYPES}")

    retain_drift_state = shift_type == "concept"
    train = simulate_dataset(
        n_train,
        base_seed_train + replication,
        g_type,
        retain_drift_state=retain_drift_state,
        **dgp_kwargs,
    )
    test_base = simulate_dataset(
        n_test,
        base_seed_test + replication,
        g_type,
        retain_drift_state=retain_drift_state,
        **dgp_kwargs,
    )

    if shift_type == "none":
        return {"train": train, "test": test_base}

    environment_seeds = (
        base_seed_test + 10000 * (replication + 1) + j
        for j in range(num_environments)
    )
    if shift_type == "covariate":
        environments, weights, divergences = [], [], []
        for seed in environment_seeds:
            environment, q, realized_kl = apply_covariate_shift(
                test_base, severity=severity, seed=seed, center_z=center_z
            )
            environments.append(environment)
            weights.append(q)
            divergences.append(realized_kl)
        test = {
            "base": test_base,
            "envs": environments,
            "q_list": weights,
            "kl_list": divergences,
            "shift_eta": float(severity),
        }
    else:
        environments = [
            apply_concept_drift(
                test_base,
                severity=severity,
                seed=seed,
                gamma=float(dgp_kwargs["gamma"]),
                m0=float(dgp_kwargs["m0"]),
                delta=float(dgp_kwargs["delta"]),
                g_type=g_type,
                w1=float(dgp_kwargs["w1"]),
                w2=float(dgp_kwargs["w2"]),
                b=float(dgp_kwargs["b"]),
                R=float(dgp_kwargs["R"]),
            )
            for seed in environment_seeds
        ]
        test = {
            "base": test_base,
            "envs": environments,
            "lambda2": float(severity),
        }
    return {"train": train, "test": test}


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
    parser.add_argument(
        "--shift_type", choices=SHIFT_TYPES, default="none"
    )
    parser.add_argument("--severity", type=float, default=0.0)
    parser.add_argument(
        "--g_type", choices=("linear", "nonlinear"), default="nonlinear"
    )
    parser.add_argument("--num_rep", type=int, default=100)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=2500)
    parser.add_argument("--num_environments", type=int, default=100)
    parser.add_argument("--out_dir", type=Path, default=Path("./data/Simulation"))
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
    parser.add_argument("--center_z", action="store_true")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--no_plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)
    parser.add_argument("--plot_n", type=int, default=6000)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.severity < 0:
        raise ValueError("severity must be nonnegative")
    if args.shift_type == "none" and args.severity != 0.0:
        raise ValueError("severity must be 0 when shift_type is 'none'")
    for name in ("num_rep", "n_train", "n_test", "num_environments"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def output_stem(shift_type: str, severity: float) -> str:
    if shift_type == "none":
        return "Shift_0.0"
    prefix = "Shift" if shift_type == "covariate" else "Drift"
    return f"{prefix}_{float(severity)}"


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    output_dir = args.out_dir / args.g_type
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
    n_test = 10000 if args.shift_type == "none" else args.n_test
    stem = output_stem(args.shift_type, args.severity)

    for replication in range(args.num_rep):
        data = generate_replication(
            replication,
            n_train=args.n_train,
            n_test=n_test,
            g_type=args.g_type,
            shift_type=args.shift_type,
            severity=args.severity,
            num_environments=args.num_environments,
            center_z=args.center_z,
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
