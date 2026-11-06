"""Policy fitting, validation-only OOB selection and held-out evaluation."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from policy_core.learning import fit_candidate
from policy_core.models import Policy
from policy_core.evaluation import coefficient_grid, select_oob, validation_scores
from policy_core.robust_values import robust_values
from .dgp import (
    fitting_seed,
    held_out_data,
    oob_counts,
    shift_direction_seeds,
    shift_resampling_seed,
    training_data,
)
from . import shift

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


def fit(study, case, rep, width, method, destination, device, data=None):
    """Fit one candidate once; an existing candidate directory is never overwritten."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if data is None:
        data = training_data(study, case, rep)
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


def shift_evaluation(rep, test, predictions):
    """Minimum policy mean over 100 paired environments at each severity."""
    n = len(test["X"])
    y = test["Y_all"].astype(np.float64)
    p0, u, z, risky = (
        test[k].ravel() for k in ("tail_p0", "tail_u", "tail_z", "tail_risky_action")
    )
    baseline = {method: y[np.arange(n), pred] for method, pred in predictions.items()}
    h = shift.susceptibility(test["X"], shift_direction_seeds(rep))
    rows = []
    for severity in shift.SEVERITIES:
        weights, _, _ = shift.tilt_weights(test["X"], severity)
        for m in range(100):
            if severity == 0:
                cov = con = {
                    method: float(values.mean()) for method, values in baseline.items()
                }
            else:
                rng = np.random.default_rng(shift_resampling_seed(rep, severity, m))
                indices = rng.choice(n, n, replace=True, p=weights)
                cov = {
                    method: float(values[indices].mean())
                    for method, values in baseline.items()
                }
                p, _, _ = shift.solve_concept_probability(p0, h[m], severity)
                potential = (
                    shift.concept_outcomes(risky, u, z, p)
                    .astype(np.float32)
                    .astype(np.float64)
                )
                con = {
                    method: float(potential[np.arange(n), pred].mean())
                    for method, pred in predictions.items()
                }
            for mode, values in (("Covariate shift", cov), ("Concept drift", con)):
                rows.append(
                    dict(rep=rep, mode=mode, severity=severity, environment=m, **values)
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


def run(study, cases, reps, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    methods = ("robust",) if study == "baseline" else METHODS
    selections, selected_paths = [], {}
    for case in cases:
        for rep in reps:
            data = training_data(study, case, rep)
            validation = (
                held_out_data(case, rep, "validation")
                if study == "baseline"
                else data["validation"]
            )
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
            test = held_out_data(case, rep, "test")
            pred = predict(selected_paths[case, rep, method], WIDTHS, test["X"], device)
            rows.append(dict(**chosen, **baseline_evaluation(test, pred, chosen)))
        frame = pd.DataFrame(rows)
        summary = summarize_baseline(frame)
    else:
        environments = []
        for rep in reps:
            test = training_data(study, "moderate", rep)["test"]
            predictions = {}
            for method in methods:
                pick = next(
                    r for r in selections if r["rep"] == rep and r["method"] == method
                )
                path = selected_paths["moderate", rep, method][pick["width"] // 2 - 1]
                predictions[method] = predict(
                    [path], [pick["width"]], test["X"], device
                )[:, 0]
            environments.append(shift_evaluation(rep, test, predictions))
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
