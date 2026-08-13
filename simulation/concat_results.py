"""Summarize simulation results across replications."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np


def severity_label(value):
    """Match the labels written by run_experiment.py."""
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return str(value)


def result_path(base_dir, mode, shift, baseline, replication):
    """Return the result file for one replication."""
    if mode == "raw":
        baseline_tag = "" if baseline == "None" else f"-{baseline}"
        name = f"Shift_0.0{baseline_tag}-exp_id-{replication}-DRO.pkl"
    else:
        label = severity_label(shift)
        prefix = "Shift" if mode == "shift" else "Drift"
        name = f"{prefix}_{label}-exp_id-{replication}-min.pkl"
    return base_dir / name


def aggregate_results(base_dir, mode, shift, baseline, num_runs):
    """Load one experimental condition and return scalar results by key."""
    paths = [
        result_path(base_dir, mode, shift, baseline, replication)
        for replication in range(num_runs)
    ]
    missing = [i for i, path in enumerate(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} of {num_runs} result files; "
            f"replication IDs: {missing}"
        )

    values = defaultdict(list)
    for path in paths:
        for key, value in joblib.load(path).items():
            values[key].append(float(np.asarray(value)))
    return values


def write_summary(values, output_path):
    """Write means and sample standard deviations to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["key", "n", "mean", "std"])
        for key, observations in sorted(values.items()):
            array = np.asarray(observations, dtype=float)
            std = np.std(array, ddof=1) if array.size > 1 else 0.0
            writer.writerow(
                [key, array.size, f"{np.mean(array):.8g}", f"{std:.8g}"]
            )


def summary_filename(mode, shift, baseline):
    """Return a filename that distinguishes nominal, shift, and drift."""
    if mode == "raw":
        condition = "Nominal"
    else:
        prefix = "Shift" if mode == "shift" else "Drift"
        condition = f"{prefix}_{severity_label(shift)}"
    baseline_tag = "" if baseline == "None" else f"-{baseline}"
    return f"summary_{condition}{baseline_tag}.csv"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize simulation results across replications."
    )
    parser.add_argument(
        "--base_dir",
        default="results/Simulation",
        help="Directory containing the linear/nonlinear result folders.",
    )
    parser.add_argument(
        "--mode", default="raw", choices=["raw", "shift", "drift"]
    )
    parser.add_argument(
        "--robust_met", default="robust", choices=["robust", "non-robust"]
    )
    parser.add_argument(
        "--type", default="nonlinear", choices=["linear", "nonlinear"]
    )
    parser.add_argument(
        "--baseline", default="None", choices=["None", "SNLN", "KLLN"]
    )
    parser.add_argument(
        "--dist_shift",
        type=float,
        default=0,
        choices=[0, 0.5, 1, 1.5, 2, 2.5, 3],
    )
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--out_root", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "raw" and args.dist_shift != 0:
        raise ValueError("raw mode requires --dist_shift 0.")
    experiment_dir = Path(args.base_dir) / args.type / args.robust_met
    values = aggregate_results(
        experiment_dir,
        args.mode,
        args.dist_shift,
        args.baseline,
        args.num_runs,
    )

    output_dir = args.out_root or experiment_dir
    output_path = output_dir / summary_filename(
        args.mode, args.dist_shift, args.baseline
    )
    write_summary(values, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
