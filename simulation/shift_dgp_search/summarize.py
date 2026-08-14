"""Summarize one candidate and update the transparent attempt log."""

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from dgp import CONFIGS, SEVERITIES


FIELDS = [
    "candidate",
    "stage",
    "n",
    "severity",
    "robust_mean",
    "nonrobust_mean",
    "improvement_pct",
    "ci_low",
    "ci_high",
    "frequency",
    "threshold",
    "stable_outcome",
    "good_outcome",
    "p_low",
    "p_high",
    "risky_propensity",
]


def severity_label(severity):
    severity = float(severity)
    return str(int(severity)) if severity.is_integer() else str(severity)


def load_values(result_root, candidate, severity, method, num_reps):
    label = severity_label(severity)
    key = f"{method}_Q_min"
    directory = (
        result_root
        / candidate
        / f"nonlinear_shift={severity:.1f}"
        / method
    )
    return np.asarray(
        [
            joblib.load(directory / f"Shift_{label}-exp_id-{i}-min.pkl")[key]
            for i in range(num_reps)
        ],
        dtype=float,
    )


def summarize(args):
    config = CONFIGS[args.candidate]
    rows = []
    for severity in args.severities:
        robust = load_values(
            args.result_root, args.candidate, severity, "robust", args.num_reps
        )
        nonrobust = load_values(
            args.result_root,
            args.candidate,
            severity,
            "non-robust",
            args.num_reps,
        )
        improvement = 100.0 * (robust - nonrobust) / nonrobust
        standard_error = improvement.std(ddof=1) / np.sqrt(args.num_reps)
        critical = 4.303 if args.num_reps == 3 else (2.262 if args.num_reps == 10 else 1.984)
        row = {
            "candidate": args.candidate,
            "stage": args.stage,
            "n": args.num_reps,
            "severity": f"{severity:g}",
            "robust_mean": f"{robust.mean():.8g}",
            "nonrobust_mean": f"{nonrobust.mean():.8g}",
            "improvement_pct": f"{improvement.mean():.8g}",
            "ci_low": f"{improvement.mean() - critical * standard_error:.8g}",
            "ci_high": f"{improvement.mean() + critical * standard_error:.8g}",
            "frequency": config.frequency,
            "threshold": config.threshold,
            "stable_outcome": config.stable_outcome,
            "good_outcome": config.good_outcome,
            "p_low": config.p_low,
            "p_high": config.p_high,
            "risky_propensity": config.risky_propensity,
        }
        rows.append(row)
        print(
            f"lambda={severity:g}: robust={robust.mean():.6f}, "
            f"nonrobust={nonrobust.mean():.6f}, "
            f"improvement={improvement.mean():.3f}%",
            flush=True,
        )
    return rows


def update_attempt_log(path, new_rows):
    existing = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
    candidate = new_rows[0]["candidate"]
    stage = new_rows[0]["stage"]
    existing = [
        row
        for row in existing
        if not (row["candidate"] == candidate and row["stage"] == stage)
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing + new_rows)


def build_parser():
    search_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--num_reps", type=int, required=True)
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        choices=SEVERITIES,
        default=list(SEVERITIES),
    )
    parser.add_argument("--result_root", type=Path, default=search_dir / "results")
    parser.add_argument("--attempt_log", type=Path, default=search_dir / "attempts.csv")
    return parser


def main():
    args = build_parser().parse_args()
    rows = summarize(args)
    update_attempt_log(args.attempt_log, rows)
    print(f"Updated {args.attempt_log}")


if __name__ == "__main__":
    main()
