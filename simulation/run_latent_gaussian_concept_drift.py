"""Run the Table 2 experiment under KL-calibrated concept drift."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import basics_with_holdout as basics
import opt_policy_learn_with_holdout as opt_policy_learn
import optimization_problem_with_holdout as optimization
from generate_dgp import simulate_dataset
from latent_gaussian_concept_drift import (
    build_drift_directions,
    drifted_outcomes,
)


RESULT_COLUMNS = [
    "replication",
    "lambda_2",
    "eta_test",
    "robust_q_min",
    "nonrobust_q_min",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_dataset(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in data.items()}


def training_args(robust_method: str, training_radius: float) -> argparse.Namespace:
    """Return the Table 2 policy-training configuration."""
    return argparse.Namespace(
        type="nonlinear",
        batch_size=128,
        epochs=50,
        density_epochs=50,
        lr=0.005,
        weight_decay=0,
        num_treat=2,
        max_iter=100,
        amb_radius=training_radius,
        Prob_depth=2,
        Prob_act_list=["ReLU", "Sigmoid"],
        Prob_neurons_list=[16, 8],
        Prob_dropout_list=[0, 0],
        norm_met="normalized",
        robust_met=robust_method,
        propensity_met="known",
        opt_procedure="data-driven",
        alpha_lb=0.1,
        dist_shift=0.0,
        test_size=2500,
        optimization_epochs=1000,
        estimator_choices="normalized_estimator",
        baseline="None",
        verbose=False,
        plot_boundary=False,
        mode="drift",
    )


def policy_actions(
    pol_dict: dict,
    learned_policy: dict,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    depth = learned_policy["opt_k"]
    model_path = learned_policy[f"policy_{depth}"]
    policy = optimization.NN_Sieves(
        pol_dict,
        pol_dict["dim_X"],
        device,
        Nlayers=depth,
        Nclasses=pol_dict["num_treat"],
    )
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    with torch.no_grad():
        logits = policy(torch.as_tensor(x, dtype=torch.float32, device=device))
    return logits.argmax(dim=1).cpu().numpy()


def train_policy(
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    replication: int,
    robust_method: str,
    training_radius: float,
    device: torch.device,
) -> np.ndarray:
    args = training_args(robust_method, training_radius)
    pol_dict = basics.initialize_setting(
        device,
        copy_dataset(train_data),
        copy_dataset(test_data),
        args,
    )
    learned_policy = opt_policy_learn.policy_learning(
        pol_dict=pol_dict,
        exp_id=replication,
        device=device,
    )
    return policy_actions(pol_dict, learned_policy, test_data["X"], device)


def baseline_data(
    replication: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_data = simulate_dataset(10000, 42 + replication, "nonlinear")
    test_data = simulate_dataset(
        2500,
        1042 + replication,
        "nonlinear",
        retain_drift_state=True,
    )
    return train_data, test_data


def training_view(test_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = (
        "X",
        "T",
        "Y",
        "prop_score",
        "Y_all",
        "prop_score_all",
        "mu",
    )
    return {key: test_data[key] for key in keys}


def environment_values(
    potential_outcomes: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    action_index = np.broadcast_to(
        actions, potential_outcomes.shape[:2]
    )[:, :, None]
    selected = np.take_along_axis(
        potential_outcomes, action_index, axis=2
    ).squeeze(axis=2)
    return selected.mean(axis=1)


def aggregate_results(raw_path: Path, table_path: Path) -> None:
    results = pd.read_csv(raw_path)
    rows = []
    for (lambda_2, eta_test), group in results.groupby(
        ["lambda_2", "eta_test"], sort=False
    ):
        robust = group["robust_q_min"]
        nonrobust = group["nonrobust_q_min"]
        robust_mean = robust.mean()
        nonrobust_mean = nonrobust.mean()
        rows.append(
            {
                "lambda_2": lambda_2,
                "eta_test": eta_test,
                "robust_mean": robust_mean,
                "robust_se": robust.sem(),
                "nonrobust_mean": nonrobust_mean,
                "nonrobust_se": nonrobust.sem(),
                "improvement_percent": (
                    100.0 * (robust_mean - nonrobust_mean) / nonrobust_mean
                ),
            }
        )
    pd.DataFrame(rows).to_csv(table_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Table 2 KL-calibrated concept-drift experiment."
    )
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--n-replications", type=int, default=100)
    parser.add_argument("--n-drift-environments", type=int, default=100)
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    parser.add_argument("--training-radius", type=float, default=0.2)
    parser.add_argument("--shift-scale", type=float, default=0.2)
    parser.add_argument("--heterogeneity-scale", type=float, default=0.35)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/latent_gaussian_concept_drift"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in ("robust", "non-robust"):
        (output_dir / "models" / "Simulation" / "nonlinear" / method).mkdir(
            parents=True, exist_ok=True
        )

    raw_path = output_dir / "replication_results.csv"
    table_path = output_dir / "table2_concept_drift.csv"

    print("lambda_2 -> eta_test")
    for value in args.lambda_grid:
        eta_test = 0.5 * (args.shift_scale * value) ** 2
        print(f"{value:g} -> {eta_test:.3f}")

    original_dir = Path.cwd()
    os.chdir(output_dir)
    try:
        replications = range(
            args.start_rep, args.start_rep + args.n_replications
        )
        robust_actions = {}
        set_seed(1)
        print("\nTraining robust policies")
        for replication in replications:
            print(f"  replication {replication}")
            train_data, test_data = baseline_data(replication)
            robust_actions[replication] = train_policy(
                train_data,
                training_view(test_data),
                replication,
                "robust",
                args.training_radius,
                device,
            )

        set_seed(1)
        print("\nTraining non-robust policies and evaluating drift")
        with raw_path.open("w", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()

            for replication in replications:
                print(
                    f"\nReplication {replication + 1 - args.start_rep}"
                    f"/{args.n_replications} (id={replication})"
                )
                train_data, test_data = baseline_data(replication)
                nonrobust_actions = train_policy(
                    train_data,
                    training_view(test_data),
                    replication,
                    "non-robust",
                    args.training_radius,
                    device,
                )

                u_base = np.tanh(6.0 * test_data["score"].reshape(-1))
                directions = build_drift_directions(
                    test_data["X"],
                    u_base,
                    args.n_drift_environments,
                    seed=args.base_seed,
                    heterogeneity_scale=args.heterogeneity_scale,
                )

                for lambda_2 in args.lambda_grid:
                    outcomes, eta_test = drifted_outcomes(
                        test_data["mu"],
                        test_data["eps_all"],
                        directions,
                        lambda_2,
                        shift_scale=args.shift_scale,
                    )
                    robust_q_min = environment_values(
                        outcomes, robust_actions[replication]
                    ).min()
                    nonrobust_q_min = environment_values(
                        outcomes, nonrobust_actions
                    ).min()
                    writer.writerow(
                        {
                            "replication": replication,
                            "lambda_2": lambda_2,
                            "eta_test": eta_test,
                            "robust_q_min": robust_q_min,
                            "nonrobust_q_min": nonrobust_q_min,
                        }
                    )
                    print(
                        f"  lambda_2={lambda_2:g}: "
                        f"robust={robust_q_min:.6f}, "
                        f"non-robust={nonrobust_q_min:.6f}"
                    )
                result_file.flush()
    finally:
        os.chdir(original_dir)

    aggregate_results(raw_path, table_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {table_path}")


if __name__ == "__main__":
    main()
