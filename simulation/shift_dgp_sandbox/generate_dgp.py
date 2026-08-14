"""Generate the isolated X-dependent covariate-shift experiment."""

import argparse
from pathlib import Path

import joblib
import numpy as np


SEVERITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SHIFT_SCALE = 0.20

Y_BAD = 0.0
Y_GOOD = 0.95
Y_STABLE = 0.79


def sample_uniform_ball(n, d, rng):
    """Sample uniformly from the unit Euclidean ball in R^d."""
    directions = rng.normal(size=(n, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return directions * radii


def sample_actions(probabilities, rng):
    """Sample one action from each row of a probability matrix."""
    return np.array(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=np.int64,
    )


def sigmoid(value):
    """Compute a numerically stable sigmoid."""
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def nonlinear_boundary(x2):
    """Return the S-shaped decision boundary used in the main simulation."""
    boundary = np.empty_like(x2, dtype=np.float64)
    upper = x2 >= 0
    boundary[upper] = np.sqrt(
        np.maximum(0.25 - (x2[upper] - 0.5) ** 2, 0.0)
    )
    boundary[~upper] = -np.sqrt(
        np.maximum(0.25 - (x2[~upper] + 0.5) ** 2, 0.0)
    )
    return boundary


def vulnerability_score(x):
    """Return the fixed lower-tail-risk surface."""
    z = 1.4 * x[:, 2] - 0.9 * x[:, 3] + 0.8 * x[:, 2] * x[:, 4] - 0.35
    return sigmoid(8.0 * z)


def simulate_dataset(n, seed, *, weak_overlap=False):
    """Generate logged outcomes with X-dependent lower-tail risk."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    vulnerability = vulnerability_score(x)

    score = x[:, 0] - nonlinear_boundary(x[:, 1])
    risky_action = (score >= 0).astype(np.int64)
    catastrophe_probability = 0.04 + 0.20 * vulnerability
    risky_outcome = np.where(
        rng.uniform(size=n) < catastrophe_probability,
        Y_BAD,
        Y_GOOD,
    )

    outcomes = np.full((n, 2), Y_STABLE, dtype=np.float64)
    outcomes[np.arange(n), risky_action] = risky_outcome

    expected_outcomes = np.full((n, 2), Y_STABLE, dtype=np.float64)
    expected_outcomes[np.arange(n), risky_action] = (
        (1.0 - catastrophe_probability) * Y_GOOD
    )

    risky_propensity = 0.75 + (0.20 * vulnerability if weak_overlap else 0.0)
    propensity = np.empty((n, 2), dtype=np.float64)
    propensity[np.arange(n), risky_action] = risky_propensity
    propensity[np.arange(n), 1 - risky_action] = 1.0 - risky_propensity

    action = sample_actions(propensity, rng)
    rows = np.arange(n)
    return {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1),
        "Y": outcomes[rows, action].reshape(-1, 1).astype(np.float32),
        "prop_score": propensity[rows, action].reshape(-1, 1).astype(np.float32),
        "Y_all": outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": expected_outcomes.astype(np.float32),
        "vulnerability": vulnerability.reshape(-1, 1).astype(np.float32),
    }


def kl_to_uniform(weights):
    """Compute KL(weights || uniform)."""
    n = weights.size
    weights = np.clip(weights, 1e-300, 1.0)
    return float(np.sum(weights * (np.log(weights) + np.log(n))))


def tilt_weights(score, beta):
    """Return normalized exponential-tilting weights."""
    logits = float(beta) * np.asarray(score, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def calibrated_weights(score, severity):
    """Calibrate the tilt to 0.5 * (0.2 * severity)^2 KL."""
    target_kl = 0.5 * (SHIFT_SCALE * float(severity)) ** 2
    lower, upper = 0.0, 1.0
    while kl_to_uniform(tilt_weights(score, upper)) < target_kl:
        upper *= 2.0

    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if kl_to_uniform(tilt_weights(score, midpoint)) < target_kl:
            lower = midpoint
        else:
            upper = midpoint

    weights = tilt_weights(score, 0.5 * (lower + upper))
    return weights, kl_to_uniform(weights)


def generate_replication(args, replication):
    """Generate one shared train/test base and all shifted resampling indices."""
    train = simulate_dataset(
        args.n_train,
        args.base_seed_train + replication,
        weak_overlap=args.variant == "fallback",
    )
    test_base = simulate_dataset(
        args.n_test,
        args.base_seed_test + replication,
    )

    score = test_base["vulnerability"].reshape(-1)
    shifts = {}
    for severity_index, severity in enumerate(SEVERITIES):
        weights, realized_kl = calibrated_weights(score, severity)
        indices = []
        for environment in range(args.num_environments):
            seed = (
                args.base_seed_test
                + 10000 * (replication + 1)
                + 1000 * severity_index
                + environment
            )
            rng = np.random.default_rng(seed)
            indices.append(
                rng.choice(
                    args.n_test,
                    size=args.n_test,
                    replace=True,
                    p=weights,
                )
            )
        shifts[f"{severity:g}"] = {
            "indices": np.asarray(indices, dtype=np.int32),
            "weights": weights,
            "kl": realized_kl,
        }

    return {
        "variant": args.variant,
        "train": train,
        "test_base": test_base,
        "shifts": shifts,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["primary", "fallback"], default="primary")
    parser.add_argument("--start_rep", type=int, default=0)
    parser.add_argument("--num_reps", type=int, default=10)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=2500)
    parser.add_argument("--num_environments", type=int, default=100)
    parser.add_argument("--base_seed_train", type=int, default=42)
    parser.add_argument("--base_seed_test", type=int, default=1042)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = args.out_dir / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    for replication in range(args.start_rep, args.start_rep + args.num_reps):
        output_path = output_dir / f"exp-{replication}.pkl"
        joblib.dump(generate_replication(args, replication), output_path)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
