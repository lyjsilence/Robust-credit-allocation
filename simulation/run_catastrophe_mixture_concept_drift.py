"""Run both Table 2 panels under the lower-tail-risk DGP."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

import basics_with_holdout as basics
import opt_policy_learn_with_holdout as policy_learning
import optimization_problem_with_holdout as optimization
from catastrophe_mixture_concept_drift import (
    Y_STABLE,
    build_susceptibility,
    covariate_environment_values,
    drifted_outcomes,
    environment_probabilities,
    population_worst_risky_value,
    simulate_catastrophe_dataset,
)


RESULT_COLUMNS = [
    "replication",
    "lambda_2",
    "eta_test",
    "robust_q_min",
    "nonrobust_q_min",
    "robust_risky_exposure",
    "nonrobust_risky_exposure",
]

COVARIATE_RESULT_COLUMNS = [
    "replication",
    "lambda_1",
    "mean_realized_kl",
    "robust_q_min",
    "nonrobust_q_min",
    "robust_risky_exposure",
    "nonrobust_risky_exposure",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_args(method: str, radius: float) -> argparse.Namespace:
    """Return the original Table 2 training configuration."""
    return argparse.Namespace(
        type="nonlinear",
        batch_size=128,
        epochs=50,
        density_epochs=50,
        lr=0.005,
        weight_decay=0,
        num_treat=2,
        max_iter=100,
        amb_radius=radius,
        Prob_depth=2,
        Prob_act_list=["ReLU", "Sigmoid"],
        Prob_neurons_list=[16, 8],
        Prob_dropout_list=[0, 0],
        norm_met="normalized",
        robust_met=method,
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


def copy_dataset(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in data.items()}


def training_view(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = (
        "X",
        "T",
        "Y",
        "prop_score",
        "Y_all",
        "prop_score_all",
        "mu",
    )
    return {key: data[key] for key in keys}


def policy_actions(
    pol_dict: dict,
    learned_policy: dict,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    depth = learned_policy["opt_k"]
    policy = optimization.NN_Sieves(
        pol_dict,
        pol_dict["dim_X"],
        device,
        Nlayers=depth,
        Nclasses=2,
    )
    policy.load_state_dict(
        torch.load(learned_policy[f"policy_{depth}"], map_location=device)
    )
    policy.eval()
    with torch.no_grad():
        logits = policy(torch.as_tensor(x, dtype=torch.float32, device=device))
    return logits.argmax(1).cpu().numpy()


def train_policy(
    train_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    replication: int,
    method: str,
    radius: float,
    device: torch.device,
) -> np.ndarray:
    args = training_args(method, radius)
    pol_dict = basics.initialize_setting(
        device,
        copy_dataset(train_data),
        copy_dataset(training_view(test_data)),
        args,
    )
    learned = policy_learning.policy_learning(pol_dict, replication, device)
    return policy_actions(pol_dict, learned, test_data["X"], device)


def baseline_data(
    replication: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = simulate_catastrophe_dataset(10000, 42 + replication)
    test = simulate_catastrophe_dataset(
        2500, 1042 + replication, retain_drift_state=True
    )
    return train, test


def environment_values(
    outcomes: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    action_index = np.broadcast_to(actions, outcomes.shape[:2])[:, :, None]
    selected = np.take_along_axis(
        outcomes, action_index, axis=2
    ).squeeze(2)
    return selected.mean(1)


def manipulation_metrics(
    test_data: dict[str, np.ndarray],
    robust_actions: np.ndarray,
    nonrobust_actions: np.ndarray,
) -> dict[str, float]:
    risky = test_data["risky_action"].reshape(-1)
    stable = 1 - risky
    rows = np.arange(risky.size)
    risky_mean = test_data["Y_all"][rows, risky].mean()
    stable_mean = test_data["Y_all"][rows, stable].mean()
    return {
        "risky_mean_advantage": float(risky_mean - stable_mean),
        "population_worst_stable_advantage": float(
            Y_STABLE - population_worst_risky_value(0.20)
        ),
        "robust_risky_exposure": float(
            np.mean(robust_actions == risky)
        ),
        "nonrobust_risky_exposure": float(
            np.mean(nonrobust_actions == risky)
        ),
    }


def aggregate_results(
    raw_path: Path,
    table_path: Path,
    severity_column: str,
) -> None:
    results = pd.read_csv(raw_path)
    rows = []
    for severity, group in results.groupby(severity_column, sort=False):
        robust = group["robust_q_min"]
        nonrobust = group["nonrobust_q_min"]
        robust_mean = robust.mean()
        nonrobust_mean = nonrobust.mean()
        row = {
            severity_column: severity,
            "robust_mean": robust_mean,
            "robust_se": robust.sem(),
            "nonrobust_mean": nonrobust_mean,
            "nonrobust_se": nonrobust.sem(),
            "improvement_percent": (
                100.0 * (robust_mean - nonrobust_mean) / nonrobust_mean
            ),
        }
        for column in ("eta_test", "mean_realized_kl"):
            if column in group:
                row[column] = group[column].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(table_path, index=False)


def combine_table2(
    covariate_path: Path,
    concept_path: Path,
    output_path: Path,
) -> None:
    covariate = pd.read_csv(covariate_path).rename(
        columns={"lambda_1": "lambda"}
    )
    covariate.insert(0, "shift_type", "covariate")
    concept = pd.read_csv(concept_path).rename(
        columns={"lambda_2": "lambda"}
    )
    concept.insert(0, "shift_type", "concept")
    pd.concat([covariate, concept], ignore_index=True).to_csv(
        output_path, index=False
    )


def summarize_checks(
    check_path: Path,
    kl_path: Path,
    summary_path: Path,
) -> None:
    checks = pd.read_csv(check_path)
    kl_checks = pd.read_csv(kl_path)
    exposure_gap = (
        checks["nonrobust_risky_exposure"].mean()
        - checks["robust_risky_exposure"].mean()
    )
    max_kl_error = kl_checks["kl_error"].max()
    summary = pd.DataFrame(
        [
            {
                "risky_mean_advantage": checks[
                    "risky_mean_advantage"
                ].mean(),
                "population_worst_stable_advantage": checks[
                    "population_worst_stable_advantage"
                ].mean(),
                "robust_risky_exposure": checks[
                    "robust_risky_exposure"
                ].mean(),
                "nonrobust_risky_exposure": checks[
                    "nonrobust_risky_exposure"
                ].mean(),
                "exposure_gap": exposure_gap,
                "max_kl_error": max_kl_error,
                "mean_check_pass": checks[
                    "risky_mean_advantage"
                ].mean()
                >= 0.01,
                "worst_case_check_pass": checks[
                    "population_worst_stable_advantage"
                ].mean()
                >= 0.03,
                "kl_check_pass": max_kl_error < 1e-8,
                "exposure_check_pass": exposure_gap >= 0.05,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    print("\nManipulation checks")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run both Table 2 shifts under the lower-tail-risk DGP."
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
        default=Path("results/catastrophe_mixture_concept_drift"),
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
    covariate_raw_path = output_dir / "covariate_replication_results.csv"
    covariate_table_path = output_dir / "table2_covariate_shift.csv"
    combined_table_path = output_dir / "table2.csv"
    check_path = output_dir / "manipulation_checks.csv"
    kl_path = output_dir / "kl_checks.csv"
    covariate_kl_path = output_dir / "covariate_kl_checks.csv"
    summary_path = output_dir / "manipulation_summary.csv"

    print("lambda_2 -> eta_test")
    for value in args.lambda_grid:
        print(f"{value:g} -> {0.5 * (args.shift_scale * value) ** 2:.3f}")

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
                test_data,
                replication,
                "robust",
                args.training_radius,
                device,
            )

        set_seed(1)
        print("\nTraining non-robust policies and evaluating shifts")
        with raw_path.open(
            "w", newline="", encoding="utf-8"
        ) as result_file, covariate_raw_path.open(
            "w", newline="", encoding="utf-8"
        ) as covariate_file, covariate_kl_path.open(
            "w", newline="", encoding="utf-8"
        ) as covariate_kl_file, check_path.open(
            "w", newline="", encoding="utf-8"
        ) as check_file, kl_path.open(
            "w", newline="", encoding="utf-8"
        ) as kl_file:
            result_writer = csv.DictWriter(
                result_file, fieldnames=RESULT_COLUMNS
            )
            result_writer.writeheader()
            covariate_writer = csv.DictWriter(
                covariate_file, fieldnames=COVARIATE_RESULT_COLUMNS
            )
            covariate_writer.writeheader()
            check_writer = None
            kl_writer = csv.DictWriter(
                kl_file,
                fieldnames=[
                    "replication",
                    "lambda_2",
                    "environment",
                    "eta_test",
                    "realized_kl",
                    "kl_error",
                ],
            )
            kl_writer.writeheader()
            covariate_kl_writer = csv.DictWriter(
                covariate_kl_file,
                fieldnames=[
                    "replication",
                    "lambda_1",
                    "environment",
                    "realized_kl",
                ],
            )
            covariate_kl_writer.writeheader()

            for replication in replications:
                print(
                    f"\nReplication {replication + 1 - args.start_rep}"
                    f"/{args.n_replications} (id={replication})"
                )
                train_data, test_data = baseline_data(replication)
                nonrobust_actions = train_policy(
                    train_data,
                    test_data,
                    replication,
                    "non-robust",
                    args.training_radius,
                    device,
                )
                metrics = {
                    "replication": replication,
                    **manipulation_metrics(
                        test_data,
                        robust_actions[replication],
                        nonrobust_actions,
                    ),
                }
                if check_writer is None:
                    check_writer = csv.DictWriter(
                        check_file, fieldnames=list(metrics)
                    )
                    check_writer.writeheader()
                check_writer.writerow(metrics)

                susceptibility = build_susceptibility(
                    test_data["X"],
                    args.n_drift_environments,
                    seed=args.base_seed,
                    heterogeneity_scale=args.heterogeneity_scale,
                )
                for lambda_2 in args.lambda_grid:
                    probabilities, eta_test, realized_kl = (
                        environment_probabilities(
                            susceptibility,
                            lambda_2,
                            shift_scale=args.shift_scale,
                        )
                    )
                    for environment, value in enumerate(realized_kl):
                        kl_writer.writerow(
                            {
                                "replication": replication,
                                "lambda_2": lambda_2,
                                "environment": environment,
                                "eta_test": eta_test,
                                "realized_kl": value,
                                "kl_error": abs(value - eta_test),
                            }
                        )

                    outcomes = drifted_outcomes(
                        test_data["catastrophe_uniform"],
                        test_data["risky_action"],
                        probabilities,
                    )
                    robust_q_min = environment_values(
                        outcomes, robust_actions[replication]
                    ).min()
                    nonrobust_q_min = environment_values(
                        outcomes, nonrobust_actions
                    ).min()
                    result_writer.writerow(
                        {
                            "replication": replication,
                            "lambda_2": lambda_2,
                            "eta_test": eta_test,
                            "robust_q_min": robust_q_min,
                            "nonrobust_q_min": nonrobust_q_min,
                            "robust_risky_exposure": metrics[
                                "robust_risky_exposure"
                            ],
                            "nonrobust_risky_exposure": metrics[
                                "nonrobust_risky_exposure"
                            ],
                        }
                    )
                    print(
                        f"  lambda_2={lambda_2:g}: "
                        f"robust={robust_q_min:.6f}, "
                        f"non-robust={nonrobust_q_min:.6f}"
                    )

                policy_actions = np.stack(
                    [robust_actions[replication], nonrobust_actions]
                )
                for lambda_1 in args.lambda_grid:
                    values, divergences = covariate_environment_values(
                        test_data["Y_all"],
                        policy_actions,
                        lambda_1,
                        args.n_drift_environments,
                        replication,
                    )
                    for environment, value in enumerate(divergences):
                        covariate_kl_writer.writerow(
                            {
                                "replication": replication,
                                "lambda_1": lambda_1,
                                "environment": environment,
                                "realized_kl": value,
                            }
                        )

                    robust_q_min, nonrobust_q_min = values.min(axis=1)
                    covariate_writer.writerow(
                        {
                            "replication": replication,
                            "lambda_1": lambda_1,
                            "mean_realized_kl": divergences.mean(),
                            "robust_q_min": robust_q_min,
                            "nonrobust_q_min": nonrobust_q_min,
                            "robust_risky_exposure": metrics[
                                "robust_risky_exposure"
                            ],
                            "nonrobust_risky_exposure": metrics[
                                "nonrobust_risky_exposure"
                            ],
                        }
                    )
                    print(
                        f"  lambda_1={lambda_1:g}: "
                        f"robust={robust_q_min:.6f}, "
                        f"non-robust={nonrobust_q_min:.6f}"
                    )
                result_file.flush()
                covariate_file.flush()
                covariate_kl_file.flush()
                check_file.flush()
                kl_file.flush()
    finally:
        os.chdir(original_dir)

    aggregate_results(raw_path, table_path, "lambda_2")
    aggregate_results(
        covariate_raw_path, covariate_table_path, "lambda_1"
    )
    combine_table2(covariate_table_path, table_path, combined_table_path)
    summarize_checks(check_path, kl_path, summary_path)
    print(f"\nSaved {combined_table_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
