"""Summarize one unified tail-risk experiment."""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from dgp import SEVERITIES


def label(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def load_values(root, mode, severity, method, num_reps):
    condition = (
        f"nonlinear_shift={severity:.1f}"
        if mode == "covariate"
        else f"nonlinear_drift={severity:.1f}"
    )
    prefix = "Shift" if mode == "covariate" else "Drift"
    directory = root / mode / condition / method
    key = f"{method}_Q_min"
    return np.asarray(
        [
            joblib.load(
                directory / f"{prefix}_{label(severity)}-exp_id-{i}-min.pkl"
            )[key]
            for i in range(num_reps)
        ],
        dtype=float,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("covariate", "concept"), required=True)
    parser.add_argument("--num_reps", type=int, required=True)
    parser.add_argument(
        "--severities", nargs="+", type=float, choices=SEVERITIES,
        default=list(SEVERITIES)
    )
    parser.add_argument(
        "--result_root", type=Path,
        default=Path(__file__).resolve().parent / "results"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "summary.csv"
    )
    args = parser.parse_args()

    critical = 4.303 if args.num_reps == 3 else (2.262 if args.num_reps == 10 else 1.984)
    rows = []
    for severity in args.severities:
        robust = load_values(
            args.result_root, args.mode, severity, "robust", args.num_reps
        )
        nonrobust = load_values(
            args.result_root, args.mode, severity, "non-robust", args.num_reps
        )
        paired = 100.0 * (robust - nonrobust) / nonrobust
        standard_error = paired.std(ddof=1) / np.sqrt(args.num_reps)
        row = {
            "mode": args.mode,
            "n": args.num_reps,
            "severity": f"{severity:g}",
            "robust_mean": robust.mean(),
            "nonrobust_mean": nonrobust.mean(),
            "improvement_of_means": 100.0 * (robust.mean() - nonrobust.mean()) / nonrobust.mean(),
            "paired_improvement": paired.mean(),
            "ci_low": paired.mean() - critical * standard_error,
            "ci_high": paired.mean() + critical * standard_error,
        }
        rows.append(row)
        print(
            f"lambda={severity:g}: robust={robust.mean():.6f}, "
            f"nonrobust={nonrobust.mean():.6f}, "
            f"improvement={row['improvement_of_means']:.3f}%",
            flush=True,
        )

    existing = []
    if args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
    existing = [row for row in existing if row["mode"] != args.mode or int(row["n"]) != args.num_reps]
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(existing + rows)


if __name__ == "__main__":
    main()
