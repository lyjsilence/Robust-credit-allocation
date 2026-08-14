"""Summarize completed unified-sandbox replications."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def mean_se(frame, groups):
    return (
        frame.groupby(groups, sort=False)["value"]
        .agg(n="count", mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(x.count()))
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox_dir", type=Path, default=HERE)
    args = parser.parse_args()
    paths = sorted((args.sandbox_dir / "results").glob("rep-*.pkl"))
    records = [joblib.load(path) for path in paths]
    if not records:
        raise FileNotFoundError("No completed result files found")

    table1 = pd.DataFrame(row for record in records for row in record["table1"])
    table2 = pd.DataFrame(
        {**row, "replication": record["replication"]}
        for record in records for row in record["table2"]
    )
    table3 = pd.DataFrame(row for record in records for row in record["table3"])

    summary1 = mean_se(table1, ["boundary", "model", "radius"])
    summary2_long = mean_se(table2, ["mode", "severity", "method"])
    summary2 = summary2_long.pivot(
        index=["mode", "severity"], columns="method", values=["n", "mean", "se"]
    )
    summary2.columns = [f"{method.lower().replace('-', '')}_{stat}" for stat, method in summary2.columns]
    summary2 = summary2.reset_index()
    summary2["improvement"] = 100.0 * (
        summary2["robust_mean"] - summary2["nonrobust_mean"]
    ) / summary2["nonrobust_mean"]
    paired = table2.pivot(
        index=["replication", "mode", "severity"], columns="method", values="value"
    ).reset_index()
    paired["difference"] = paired["Robust"] - paired["Non-robust"]
    paired_summary = mean_se(paired.rename(columns={"difference": "value"}), ["mode", "severity"])
    paired_summary = paired_summary.rename(
        columns={"mean": "paired_difference", "se": "paired_difference_se"}
    )[["mode", "severity", "paired_difference", "paired_difference_se"]]
    summary2 = summary2.merge(paired_summary, on=["mode", "severity"])
    summary3 = mean_se(table3, ["complexity", "radius"])
    depths = pd.Series([record["selected_depth"] for record in records]).value_counts().sort_index()
    depth_summary = depths.rename_axis("selected_depth").reset_index(name="count")

    summary1.to_csv(args.sandbox_dir / "table1.csv", index=False)
    summary2.to_csv(args.sandbox_dir / "table2.csv", index=False)
    summary3.to_csv(args.sandbox_dir / "table3.csv", index=False)
    depth_summary.to_csv(args.sandbox_dir / "selected_depths.csv", index=False)
    print(f"Summarized {len(records)} replications")


if __name__ == "__main__":
    main()
