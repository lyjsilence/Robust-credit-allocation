"""Generate covariate shift and concept drift in one tail-risk framework."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np


SEVERITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SHIFT_SCALE = 0.20


@dataclass(frozen=True)
class DGPConfig:
    stable_outcome: float = 0.79
    good_outcome: float = 0.90
    bad_outcome: float = 0.00
    risky_propensity: float = 0.75
    concept_baseline: float = 0.10
    risk_frequency: int = 2
    risk_threshold: float = 0.70
    risk_sharpness: float = 30.0
    risk_low: float = 0.02
    risk_high: float = 0.78


CONFIG = DGPConfig()


def sample_uniform_ball(n, d, rng):
    directions = rng.normal(size=(n, d))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    radii = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return directions * radii


def sample_actions(probabilities, rng):
    return np.asarray(
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
    boundary[upper] = np.sqrt(np.maximum(0.25 - (x2[upper] - 0.5) ** 2, 0.0))
    boundary[~upper] = -np.sqrt(np.maximum(0.25 - (x2[~upper] + 0.5) ** 2, 0.0))
    return boundary


def risk_score(x, config=CONFIG):
    wave = np.sin(config.risk_frequency * np.pi * x[:, 2]) * np.sin(
        config.risk_frequency * np.pi * x[:, 3]
    )
    return sigmoid(config.risk_sharpness * (wave - config.risk_threshold))


def baseline_probability(x, mode, config=CONFIG):
    if mode == "concept":
        return np.full(x.shape[0], config.concept_baseline)
    score = risk_score(x, config)
    return config.risk_low + (config.risk_high - config.risk_low) * score


def simulate_dataset(n, seed, mode, config=CONFIG):
    rng = np.random.default_rng(seed)
    x = sample_uniform_ball(n, 5, rng)
    risky_action = (x[:, 0] - nonlinear_boundary(x[:, 1]) >= 0).astype(np.int64)
    probability = baseline_probability(x, mode, config)

    catastrophe_uniform = rng.uniform(size=n)
    risky_outcome = np.where(
        catastrophe_uniform < probability,
        config.bad_outcome,
        config.good_outcome,
    )
    outcomes = np.full((n, 2), config.stable_outcome)
    outcomes[np.arange(n), risky_action] = risky_outcome

    expected = np.full((n, 2), config.stable_outcome)
    expected[np.arange(n), risky_action] = (
        probability * config.bad_outcome
        + (1.0 - probability) * config.good_outcome
    )

    propensity = np.full((n, 2), 1.0 - config.risky_propensity)
    propensity[np.arange(n), risky_action] = config.risky_propensity
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
        "risky_action": risky_action.reshape(-1, 1),
        "catastrophe_uniform": catastrophe_uniform.reshape(-1, 1),
        "catastrophe_probability": probability.reshape(-1, 1).astype(np.float32),
    }


def kl_to_uniform(weights):
    weights = np.clip(weights, 1e-300, 1.0)
    return float(np.sum(weights * (np.log(weights) + np.log(weights.size))))


def tilt_weights(score, beta):
    logits = float(beta) * np.asarray(score, dtype=np.float64)
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
    return probability * np.log(probability / baseline) + (
        1.0 - probability
    ) * np.log((1.0 - probability) / (1.0 - baseline))


def solve_concept_probability(susceptibility, target, baseline):
    baseline_logit = np.log(baseline / (1.0 - baseline))

    def probability(tau):
        return sigmoid(baseline_logit + tau * susceptibility)

    lower, upper = 0.0, 1.0
    while bernoulli_kl(probability(upper), baseline).mean() < target:
        upper *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        realized = bernoulli_kl(probability(midpoint), baseline).mean()
        if realized < target:
            lower = midpoint
        else:
            upper = midpoint
    return probability(0.5 * (lower + upper))


def concept_probabilities(susceptibility, severity, baseline=CONFIG.concept_baseline):
    target = 0.5 * (SHIFT_SCALE * float(severity)) ** 2
    probability = np.stack(
        [
            solve_concept_probability(row, target, baseline)
            for row in susceptibility
        ]
    )
    return probability, bernoulli_kl(probability, baseline).mean(axis=1)


def generate_replication(args, replication):
    train = simulate_dataset(args.n_train, args.base_seed_train + replication, args.mode)
    test_base = simulate_dataset(args.n_test, args.base_seed_test + replication, args.mode)
    shifts = {}

    if args.mode == "covariate":
        score = risk_score(test_base["X"])
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
                indices.append(
                    np.random.default_rng(seed).choice(
                        args.n_test, size=args.n_test, replace=True, p=weights
                    )
                )
            shifts[f"{severity:g}"] = {
                "indices": np.asarray(indices, dtype=np.int32),
                "kl": realized_kl,
            }
    else:
        susceptibility = build_susceptibility(test_base["X"], args.num_environments)
        for severity in SEVERITIES:
            probability, realized_kl = concept_probabilities(susceptibility, severity)
            shifts[f"{severity:g}"] = {
                "probabilities": probability,
                "kl": realized_kl,
            }

    return {
        "mode": args.mode,
        "config": asdict(CONFIG),
        "train": train,
        "test_base": test_base,
        "shifts": shifts,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("covariate", "concept"), required=True)
    parser.add_argument("--start_rep", type=int, default=0)
    parser.add_argument("--num_reps", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=2500)
    parser.add_argument("--num_environments", type=int, default=100)
    parser.add_argument("--base_seed_train", type=int, default=42)
    parser.add_argument("--base_seed_test", type=int, default=1042)
    parser.add_argument(
        "--out_dir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = args.out_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    for replication in range(args.start_rep, args.start_rep + args.num_reps):
        path = output_dir / f"exp-{replication}.pkl"
        joblib.dump(generate_replication(args, replication), path)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
