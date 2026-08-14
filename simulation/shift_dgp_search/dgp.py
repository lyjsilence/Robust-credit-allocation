"""Candidate DGPs for the isolated covariate-shift search."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np


SEVERITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SHIFT_SCALE = 0.20


@dataclass(frozen=True)
class DGPConfig:
    frequency: int
    threshold: float
    sharpness: float = 20.0
    stable_outcome: float = 0.50
    good_outcome: float = 0.95
    p_low: float = 0.02
    p_high: float = 0.90
    risky_propensity: float = 0.75


CONFIGS = {
    "islands_k2_t10": DGPConfig(frequency=2, threshold=0.10),
    "islands_k2_t25": DGPConfig(frequency=2, threshold=0.25),
    "islands_k4_t10": DGPConfig(frequency=4, threshold=0.10),
    "islands_k4_t25": DGPConfig(frequency=4, threshold=0.25),
    "islands_k6_t10": DGPConfig(frequency=6, threshold=0.10),
    "islands_k6_t25": DGPConfig(frequency=6, threshold=0.25),
}


def sample_uniform_ball(n, d, rng):
    directions = rng.normal(size=(n, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return directions * radii


def sample_actions(probabilities, rng):
    return np.array(
        [rng.choice(probabilities.shape[1], p=row) for row in probabilities],
        dtype=np.int64,
    )


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def nonlinear_boundary(x2):
    boundary = np.empty_like(x2, dtype=np.float64)
    upper = x2 >= 0
    boundary[upper] = np.sqrt(
        np.maximum(0.25 - (x2[upper] - 0.5) ** 2, 0.0)
    )
    boundary[~upper] = -np.sqrt(
        np.maximum(0.25 - (x2[~upper] + 0.5) ** 2, 0.0)
    )
    return boundary


def vulnerability_score(x, config):
    """Return smooth high-risk islands based only on observed X."""
    wave = np.sin(config.frequency * np.pi * x[:, 2]) * np.sin(
        config.frequency * np.pi * x[:, 3]
    )
    return sigmoid(config.sharpness * (wave - config.threshold))


def simulate_dataset(n, seed, config):
    """Generate logged outcomes under one fixed conditional mechanism."""
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    vulnerability = vulnerability_score(x, config)

    nonlinear_score = x[:, 0] - nonlinear_boundary(x[:, 1])
    risky_action = (nonlinear_score >= 0).astype(np.int64)
    catastrophe_probability = config.p_low + (
        config.p_high - config.p_low
    ) * vulnerability
    risky_outcome = np.where(
        rng.uniform(size=n) < catastrophe_probability,
        0.0,
        config.good_outcome,
    )

    outcomes = np.full((n, 2), config.stable_outcome, dtype=np.float64)
    outcomes[np.arange(n), risky_action] = risky_outcome

    expected = np.full((n, 2), config.stable_outcome, dtype=np.float64)
    expected[np.arange(n), risky_action] = (
        1.0 - catastrophe_probability
    ) * config.good_outcome

    propensity = np.empty((n, 2), dtype=np.float64)
    propensity[np.arange(n), risky_action] = config.risky_propensity
    propensity[np.arange(n), 1 - risky_action] = 1.0 - config.risky_propensity
    action = sample_actions(propensity, rng)
    rows = np.arange(n)

    return {
        "X": x.astype(np.float32),
        "T": action.reshape(-1, 1),
        "Y": outcomes[rows, action].reshape(-1, 1).astype(np.float32),
        "prop_score": propensity[rows, action].reshape(-1, 1).astype(np.float32),
        "Y_all": outcomes.astype(np.float32),
        "prop_score_all": propensity.astype(np.float32),
        "mu": expected.astype(np.float32),
        "vulnerability": vulnerability.reshape(-1, 1).astype(np.float32),
    }


def kl_to_uniform(weights):
    n = weights.size
    weights = np.clip(weights, 1e-300, 1.0)
    return float(np.sum(weights * (np.log(weights) + np.log(n))))


def tilt_weights(score, beta):
    logits = float(beta) * np.asarray(score, dtype=np.float64)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def calibrated_weights(score, severity):
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
    config = CONFIGS[args.candidate]
    train = simulate_dataset(args.n_train, args.base_seed_train + replication, config)
    test_base = simulate_dataset(args.n_test, args.base_seed_test + replication, config)
    vulnerability = test_base["vulnerability"].reshape(-1)

    shifts = {}
    for severity_index, severity in enumerate(SEVERITIES):
        weights, realized_kl = calibrated_weights(vulnerability, severity)
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
        "candidate": args.candidate,
        "config": asdict(config),
        "train": train,
        "test_base": test_base,
        "shifts": shifts,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--start_rep", type=int, default=0)
    parser.add_argument("--num_reps", type=int, default=3)
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
    output_dir = args.out_dir / args.candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    for replication in range(args.start_rep, args.start_rep + args.num_reps):
        output_path = output_dir / f"exp-{replication}.pkl"
        joblib.dump(generate_replication(args, replication), output_path)
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
