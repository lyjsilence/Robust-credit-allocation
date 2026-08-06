"""Run Table 2 under fragile-action KL-calibrated concept drift."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fragile_action_concept_drift import (
    build_fragile_directions,
    drifted_fragile_outcomes,
    simulate_fragile_dataset,
)
from run_latent_gaussian_concept_drift import (
    aggregate_results,
    environment_values,
    set_seed,
    train_policy,
)


RESULT_COLUMNS = [
    "replication",
    "lambda_2",
    "eta_test",
    "robust_q_min",
    "nonrobust_q_min",
    "robust_fragile_exposure",
    "nonrobust_fragile_exposure",
]


def baseline_data(
    replication: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = simulate_fragile_dataset(10000, 42 + replication)
    test = simulate_fragile_dataset(
        2500, 1042 + replication, retain_drift_state=True
    )
    return train, test


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


def manipulation_metrics(
    test_data: dict[str, np.ndarray],
    robust_actions: np.ndarray,
    nonrobust_actions: np.ndarray,
) -> dict[str, float]:
    fragile = test_data["fragile_action"].reshape(-1)
    stable = 1 - fragile
    rows = np.arange(fragile.size)
    y_fragile = test_data["Y_all"][rows, fragile]
    y_stable = test_data["Y_all"][rows, stable]
    return {
        "fragile_mean_advantage": float(y_fragile.mean() - y_stable.mean()),
        "stable_q10_advantage": float(
            np.quantile(y_stable, 0.10)
            - np.quantile(y_fragile, 0.10)
        ),
        "robust_fragile_exposure": float(
            np.mean(robust_actions == fragile)
        ),
        "nonrobust_fragile_exposure": float(
            np.mean(nonrobust_actions == fragile)
        ),
    }


def summarize_manipulation(check_path: Path, summary_path: Path) -> None:
    checks = pd.read_csv(check_path)
    mean_advantage = checks["fragile_mean_advantage"].mean()
    q10_advantage = checks["stable_q10_advantage"].mean()
    exposure_gap = (
        checks["nonrobust_fragile_exposure"].mean()
        - checks["robust_fragile_exposure"].mean()
    )
    summary = pd.DataFrame(
        [
            {
                "fragile_mean_advantage": mean_advantage,
                "stable_q10_advantage": q10_advantage,
                "robust_fragile_exposure": checks[
                    "robust_fragile_exposure"
                ].mean(),
                "nonrobust_fragile_exposure": checks[
                    "nonrobust_fragile_exposure"
                ].mean(),
                "exposure_gap": exposure_gap,
                "mean_check_pass": mean_advantage >= 0.02,
                "q10_check_pass": q10_advantage >= 0.05,
                "exposure_check_pass": exposure_gap >= 0.05,
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    print("\nManipulation checks")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fragile-action concept drift for Table 2."
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
        default=Path("results/fragile_action_concept_drift"),
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
    check_path = output_dir / "manipulation_checks.csv"
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
                training_view(test_data),
                replication,
                "robust",
                args.training_radius,
                device,
            )

        set_seed(1)
        print("\nTraining non-robust policies and evaluating drift")
        with raw_path.open(
            "w", newline="", encoding="utf-8"
        ) as result_file, check_path.open(
            "w", newline="", encoding="utf-8"
        ) as check_file:
            writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            check_writer = None

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

                directions = build_fragile_directions(
                    test_data["X"],
                    args.n_drift_environments,
                    seed=args.base_seed,
                    heterogeneity_scale=args.heterogeneity_scale,
                )
                robust_exposure = metrics["robust_fragile_exposure"]
                nonrobust_exposure = metrics["nonrobust_fragile_exposure"]

                for lambda_2 in args.lambda_grid:
                    outcomes, eta_test = drifted_fragile_outcomes(
                        test_data["mu"],
                        test_data["eps_all"],
                        test_data["sigma_all"],
                        test_data["fragile_action"],
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
                            "robust_fragile_exposure": robust_exposure,
                            "nonrobust_fragile_exposure": nonrobust_exposure,
                        }
                    )
                    print(
                        f"  lambda_2={lambda_2:g}: "
                        f"robust={robust_q_min:.6f}, "
                        f"non-robust={nonrobust_q_min:.6f}"
                    )
                result_file.flush()
                check_file.flush()
    finally:
        os.chdir(original_dir)

    aggregate_results(raw_path, table_path)
    summarize_manipulation(check_path, summary_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {table_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
