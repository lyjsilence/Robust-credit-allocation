import os
import argparse
import numpy as np
import joblib
from collections import defaultdict


def aggregate_one_combo(base_dir, shift, num_runs):
    """聚合某一 (type, covar, concept) 组合"""

    # key -> list of scalars
    scalars = defaultdict(list)

    found_files = 0
    for i in range(num_runs):
        if shift == 0:
            if args.baseline == 'SNLN':
                fpath = os.path.join(base_dir, f"Shift_{shift}-SNLN-exp_id-{i}-DRO.pkl")
            elif args.baseline == 'KLLN':
                fpath = os.path.join(base_dir, f"Shift_{shift}-KLLN-exp_id-{i}-DRO.pkl")
            else:
                fpath = os.path.join(base_dir, f"Shift_{shift}-exp_id-{i}-DRO.pkl")
        else:
            if args.mode == 'shift':
                fpath = os.path.join(base_dir, f"Shift_{shift}-exp_id-{i}-min.pkl")
            else:
                fpath = os.path.join(base_dir, f"Drift_{shift}-exp_id-{i}-min.pkl")
        d = joblib.load(fpath)

        found_files += 1
        for k, v in d.items():
            a = np.asarray(v)
            scalars[k].append(float(a))

    # ---------- 写出标量 CSV ----------
    import csv
    if args.baseline == 'SNLN':
        scalar_csv = os.path.join(base_dir, f"summary_Dist_{shift}-SNLN.csv")
    elif args.baseline == 'KLLN':
        scalar_csv = os.path.join(base_dir, f"summary_Dist_{shift}-KLLN.csv")
    else:
        scalar_csv = os.path.join(base_dir, f"summary_Dist_{shift}.csv")

    with open(scalar_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "n", "mean", "std"])
        for k, vals in sorted(scalars.items()):
            arr = np.asarray(vals, dtype=float)
            n = int(arr.size)
            mean = float(np.mean(arr)) if n > 0 else np.nan
            print(mean)
            # 使用样本标准差（ddof=1）；若 n=1 则回退到 0
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            w.writerow([k, n, f"{mean:.8g}", f"{std:.8g}"])


parser = argparse.ArgumentParser()
parser.add_argument("--base_dir", type=str, default="results/Simulation",
                    help="根目录：包含 exp-{type} 子目录")
parser.add_argument("--mode", type=str, default='drift', choices=['drift', 'shift'])
parser.add_argument("--robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("--type", type=str, default="nonlinear", choices=['linear', 'nonlinear'])
parser.add_argument("--baseline", type=str, default="None", choices=['None', 'SNLN', 'KLLN'])
parser.add_argument("--dist_shift", type=float, default=2.5, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument("--num_runs", type=int, default=100)
parser.add_argument("--out_root", type=str, default=None,
                    help="输出目录（默认与每个 exp-{type} 同目录）")
args = parser.parse_args()


def concat_Q_min():
    exp_dir = os.path.join(args.base_dir, args.type, args.robust_met)
    out_base = args.out_root or exp_dir  # 默认写回各自 exp-{type} 目
    os.makedirs(out_base, exist_ok=True)

    # out_dir = os.path.join(out_base, f'Shift_{shift}-exp_id-{covar}.pkl')
    aggregate_one_combo(
        base_dir=exp_dir,
        shift=args.dist_shift,
        num_runs=args.num_runs,
    )


def concat_Q_DRO():
    exp_dir = os.path.join(args.base_dir, args.type, args.robust_met)
    out_base = args.out_root or exp_dir  # 默认写回各自 exp-{type} 目
    os.makedirs(out_base, exist_ok=True)

    # out_dir = os.path.join(out_base, f'Shift_{shift}-exp_id-{covar}.pkl')
    aggregate_one_combo(
        base_dir=exp_dir,
        shift=args.dist_shift,
        num_runs=args.num_runs,
    )


if __name__ == "__main__":
    concat_Q_DRO()
