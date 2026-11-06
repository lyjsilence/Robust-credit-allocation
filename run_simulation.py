"""Generate data, fit policies and run simulation experiments."""

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import torch

from learning import Policy, fit_candidate
from evaluation import coefficient_grid, robust_values, select_oob, validation_scores
from dgp import fitting_seed, oob_counts
import dgp

WIDTHS = tuple(range(2, 257, 2))
METHODS = ("robust", "non-robust")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(study, case, rep, width, method, destination, device, data):
    """Fit one candidate once; an existing candidate directory is never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    candidate = fitting_seed(study, case, rep, width)
    set_seed(candidate)
    path = destination / "model.pt"
    history = fit_candidate(
        data["train"],
        width,
        method,
        path,
        torch.device(device),
    )
    pd.DataFrame(history).to_csv(destination / "training.csv", index=False)
    return path


def predict(paths, widths, x, device):
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    predictions = []
    for path, width in zip(paths, widths):
        policy = Policy(width).to(device)
        policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        policy.eval()
        with torch.no_grad():
            predictions.append(torch.softmax(policy(tensor), 1).argmax(1).cpu().numpy())
    return np.column_stack(predictions)


def select(study, case, rep, method, validation, decisions):
    eta = 0.2 if method == "robust" else 0.0
    counts = oob_counts(study, case, rep)
    scores = validation_scores(validation, decisions, counts, eta=eta)
    chosen, _ = select_oob(
        **scores,
        widths=WIDTHS,
        coefficients=coefficient_grid(),
        cap=128 if study == "baseline" else 256,
    )
    return chosen


def baseline_evaluation(test, decisions, selected):
    values = robust_values(
        test["Y"], test["T"], decisions, test["prop_score"], eta=0.2
    )["value"][0]
    q = float(values[selected["width"] // 2 - 1])
    oracle = float(values[:64].max())
    return dict(test_value=q, oracle_value=oracle, test_regret=oracle - q)


def shift_evaluation(rep, test, predictions, data_dir, levels, shift_mode):
    """Evaluate both policies on the same saved deployment environments."""
    n = len(test["X"])
    y = test["Y_all"].astype(np.float64)
    baseline = {method: y[np.arange(n), pred] for method, pred in predictions.items()}
    modes = ("covariate", "concept") if shift_mode == "both" else (shift_mode,)
    labels = {"covariate": "Covariate shift", "concept": "Concept drift"}
    rows = []
    for severity in levels:
        scores = {}
        for mode in modes:
            with np.load(
                dgp.environment_path(data_dir, rep, mode, severity)
            ) as archive:
                if mode == "covariate":
                    indices = archive["indices"]
                    scores[mode] = {
                        method: np.array([values[index].mean() for index in indices])
                        for method, values in baseline.items()
                    }
                else:
                    outcomes = archive["outcomes"].astype(np.float64)
                    scores[mode] = {
                        method: np.array(
                            [
                                environment[np.arange(n), pred].mean()
                                for environment in outcomes
                            ]
                        )
                        for method, pred in predictions.items()
                    }
        for m in range(100):
            for mode in modes:
                rows.append(
                    dict(
                        rep=rep,
                        mode=labels[mode],
                        severity=severity,
                        environment=m,
                        **{
                            method: float(values[m])
                            for method, values in scores[mode].items()
                        },
                    )
                )
    return pd.DataFrame(rows)


def summarize_baseline(frame):
    return (
        frame.groupby("case")
        .agg(
            n=("rep", "size"),
            median_width=("width", "median"),
            mean_q=("test_value", "mean"),
            sd_q=("test_value", "std"),
            mean_regret=("test_regret", "mean"),
            sd_regret=("test_regret", "std"),
        )
        .reset_index()
    )


def summarize_shift(frame):
    rows = []
    for (mode, severity), part in frame.groupby(["mode", "severity"]):
        n = len(part)
        a, b = part["robust"], part["non-robust"]
        rows.append(
            dict(
                mode=mode,
                severity=severity,
                n=n,
                robust_mean=a.mean(),
                robust_se=a.std(ddof=1) / np.sqrt(n),
                nonrobust_mean=b.mean(),
                nonrobust_se=b.std(ddof=1) / np.sqrt(n),
                improvement_percent=100 * (a - b).mean() / b.mean(),
            )
        )
    return pd.DataFrame(rows)


def run(
    study,
    cases,
    reps,
    output,
    device,
    data_dir,
    levels=dgp.SEVERITIES,
    shift_mode="both",
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    methods = ("robust",) if study == "baseline" else METHODS
    selections, selected_paths = [], {}
    for case in cases:
        for rep in reps:
            data = dgp.load_samples(data_dir, study, case, rep)
            validation = data["validation"]
            for method in methods:
                pool = output / f"{case}/rep_{rep:03d}/{method}"
                paths = []
                for width in WIDTHS:
                    paths.append(
                        fit(
                            study,
                            case,
                            rep,
                            width,
                            method,
                            pool / f"w{width}",
                            device,
                            data,
                        )
                    )
                    print(f"{case} rep={rep} {method} width={width} fitted", flush=True)
                decisions = predict(paths, WIDTHS, validation["X"], device)
                pick = select(study, case, rep, method, validation, decisions)
                row = dict(case=case, rep=rep, method=method, **pick)
                selections.append(row)
                selected_paths[case, rep, method] = paths
    # All choices are fixed on validation before held-out policy evaluation.
    pd.DataFrame(selections).to_csv(output / "selections.csv", index=False)
    if study == "baseline":
        rows = []
        for chosen in selections:
            case, rep, method = chosen["case"], chosen["rep"], chosen["method"]
            test = dgp.load_samples(data_dir, study, case, rep)["test"]
            pred = predict(selected_paths[case, rep, method], WIDTHS, test["X"], device)
            rows.append(dict(**chosen, **baseline_evaluation(test, pred, chosen)))
        frame = pd.DataFrame(rows)
        summary = summarize_baseline(frame)
    else:
        environments = []
        for rep in reps:
            test = dgp.load_samples(data_dir, study, "moderate", rep)["test"]
            predictions = {}
            for method in methods:
                pick = next(
                    r for r in selections if r["rep"] == rep and r["method"] == method
                )
                path = selected_paths["moderate", rep, method][pick["width"] // 2 - 1]
                predictions[method] = predict(
                    [path], [pick["width"]], test["X"], device
                )[:, 0]
            environments.append(
                shift_evaluation(rep, test, predictions, data_dir, levels, shift_mode)
            )
        environments = pd.concat(environments, ignore_index=True)
        environments.to_csv(output / "environments.csv", index=False)
        frame = (
            environments.groupby(["rep", "mode", "severity"])[list(METHODS)]
            .min()
            .reset_index()
        )
        summary = summarize_shift(frame)
    frame.to_csv(output / "replication_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("generate", "fit", "run", "summarize"), required=True
    )
    parser.add_argument("--study", choices=("baseline", "shift"), required=True)
    parser.add_argument("--cases", nargs="+", choices=dgp.CASES)
    parser.add_argument("--start-rep", type=int, default=0)
    parser.add_argument("--num-reps", type=int, default=100)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--width", type=int, default=10, help="Network width for --mode fit"
    )
    parser.add_argument(
        "--method", choices=METHODS, default="robust", help="Method for --mode fit"
    )
    parser.add_argument(
        "--shift-mode", choices=("covariate", "concept", "both"), default="both"
    )
    parser.add_argument(
        "--level",
        nargs="+",
        type=float,
        choices=dgp.SEVERITIES,
        help="Shift severities; default: all levels 0 through 12",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    cases = args.cases or (["moderate"] if args.study == "shift" else list(dgp.CASES))
    levels = sorted(set(args.level)) if args.level is not None else dgp.SEVERITIES
    output = args.output or Path("results") / args.study
    torch.set_num_threads(1)
    if args.mode == "summarize":
        frame = pd.read_csv(
            output / "replication_results.csv", float_precision="round_trip"
        )
        summary = (
            summarize_baseline(frame)
            if args.study == "baseline"
            else summarize_shift(frame)
        )
        summary.to_csv(output / "summary.csv", index=False)
        print(summary.to_string(index=False))
        return
    if args.num_reps < 1 or args.start_rep < 0 or args.start_rep + args.num_reps > 100:
        parser.error("Replication range must lie within 0,...,99")
    if len(set(cases)) != len(cases):
        parser.error("Cases must be distinct")
    if args.study == "shift" and cases != ["moderate"]:
        parser.error("The shift experiment uses --cases moderate")
    reps = range(args.start_rep, args.start_rep + args.num_reps)
    if args.mode in ("fit", "run"):
        required = [
            dgp.sample_path(args.data_dir, args.study, case, rep)
            for case in cases
            for rep in reps
        ]
        if args.mode == "run" and args.study == "shift":
            modes = (
                ("covariate", "concept")
                if args.shift_mode == "both"
                else (args.shift_mode,)
            )
            required += [
                dgp.environment_path(args.data_dir, rep, mode, level)
                for rep in reps
                for mode in modes
                for level in levels
            ]
        for path in required:
            if not path.is_file():
                parser.error(
                    f"Missing {path}; generate the requested data first with --mode generate"
                )
    if args.mode == "generate":
        dgp.generate_data(
            args.study, cases, reps, args.data_dir, levels, args.shift_mode
        )
    elif args.mode == "fit":
        if args.num_reps != 1 or len(cases) != 1 or args.width not in WIDTHS:
            parser.error(
                "fit requires one replication, one case and an even width 2–256"
            )
        data = dgp.load_samples(args.data_dir, args.study, cases[0], args.start_rep)
        path = fit(
            args.study,
            cases[0],
            args.start_rep,
            args.width,
            args.method,
            output,
            args.device,
            data,
        )
        print(f"Model saved to {path}")
    else:
        run(
            args.study,
            cases,
            reps,
            output,
            args.device,
            args.data_dir,
            levels,
            args.shift_mode,
        )


if __name__ == "__main__":
    main()
