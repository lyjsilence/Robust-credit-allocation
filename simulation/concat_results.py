"""Summarize all existing simulation result files in one CSV."""

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

CONDITION_PATTERN = re.compile(
    r"^(?P<dgp_type>linear|nonlinear)"
    r"(?:_(?P<mode>shift|drift)=(?P<severity>[0-9.]+))?$"
)
FILENAME_PATTERN = re.compile(
    r"^(?:Shift|Drift)_[^-]+"
    r"(?:-(?P<baseline>SNLN|KLLN))?"
    r"-exp_id-\d+-(?:DRO|min)\.pkl$"
)
MODE_ORDER = {"raw": 0, "shift": 1, "drift": 2}
METHOD_ORDER = {"robust": 0, "non-robust": 1}
BASELINE_ORDER = {"None": 0, "SNLN": 1, "KLLN": 2}


@dataclass(frozen=True)
class SummaryGroup:
    """Fields that identify one row of the summary table."""

    dgp_type: str
    mode: str
    severity: float
    method: str
    baseline: str
    metric: str


def parse_result_path(path, results_dir):
    """Read experimental metadata from a result directory and filename."""
    condition = path.parent.parent.name
    method = path.parent.name

    condition_match = CONDITION_PATTERN.fullmatch(condition)
    filename_match = FILENAME_PATTERN.fullmatch(path.name)
    if condition_match is None or filename_match is None:
        raise ValueError(f"Unrecognized result path: {path.relative_to(results_dir)}")
    if method not in METHOD_ORDER:
        raise ValueError(f"Unrecognized method directory: {method}")

    mode = condition_match.group("mode") or "raw"
    severity = float(condition_match.group("severity") or 0)
    baseline = filename_match.group("baseline") or "None"

    return {
        "dgp_type": condition_match.group("dgp_type"),
        "mode": mode,
        "severity": severity,
        "method": method,
        "baseline": baseline,
    }


def collect_results(results_dir):
    """Load every result file and group its scalar metrics."""
    result_files = sorted(results_dir.rglob("*.pkl"))
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")

    grouped_values = defaultdict(list)
    for path in result_files:
        metadata = parse_result_path(path, results_dir)
        for metric, value in joblib.load(path).items():
            group = SummaryGroup(metric=metric, **metadata)
            grouped_values[group].append(float(np.asarray(value).item()))

    return result_files, grouped_values


def group_sort_key(group):
    """Return a stable order for the summary rows."""
    return (
        group.dgp_type,
        MODE_ORDER[group.mode],
        group.severity,
        METHOD_ORDER[group.method],
        BASELINE_ORDER[group.baseline],
        group.metric,
    )


def write_summary(grouped_values, output_path):
    """Write counts, means, sample standard deviations, and standard errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dgp_type",
                "mode",
                "severity",
                "method",
                "baseline",
                "metric",
                "n",
                "mean",
                "std",
                "se",
            ]
        )

        for group in sorted(grouped_values, key=group_sort_key):
            values = np.asarray(grouped_values[group], dtype=float)
            std = np.std(values, ddof=1) if values.size > 1 else 0.0
            se = std / np.sqrt(values.size)
            writer.writerow(
                [
                    group.dgp_type,
                    group.mode,
                    f"{group.severity:g}",
                    group.method,
                    group.baseline,
                    group.metric,
                    values.size,
                    f"{np.mean(values):.8g}",
                    f"{std:.8g}",
                    f"{se:.8g}",
                ]
            )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize all existing simulation results in one CSV."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Result root (default: simulation/results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <results_dir>/summary_results.csv).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    results_dir = args.results_dir.resolve()
    output_path = args.output or results_dir / "summary_results.csv"

    result_files, grouped_values = collect_results(results_dir)
    write_summary(grouped_values, output_path)

    print(f"Read {len(result_files)} result files.")
    print(f"Saved {len(grouped_values)} summary rows to {output_path}")


if __name__ == "__main__":
    main()
