"""Run the unchanged policy learner on the unified tail-risk sandbox."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import torch


SIMULATION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMULATION_DIR))

import basics_with_holdout as basics
import opt_policy_learn_with_holdout as policy_tools

from dgp import CONFIG, SEVERITIES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def severity_label(severity):
    severity = float(severity)
    return str(int(severity)) if severity.is_integer() else str(severity)


def resampled_environments(base, indices):
    n = base["X"].shape[0]
    return [
        {
            key: value[index].copy()
            if isinstance(value, np.ndarray) and value.shape[0] == n
            else value
            for key, value in base.items()
        }
        for index in indices
    ]


def drifted_environments(base, probabilities):
    n = base["X"].shape[0]
    rows = np.arange(n)
    action = base["T"].reshape(-1)
    risky_action = base["risky_action"].reshape(-1)
    uniforms = base["catastrophe_uniform"].reshape(-1)
    environments = []

    for probability in probabilities:
        catastrophe = uniforms < probability
        risky_outcome = np.where(
            catastrophe, CONFIG.bad_outcome, CONFIG.good_outcome
        )
        outcomes = np.full((n, 2), CONFIG.stable_outcome)
        outcomes[rows, risky_action] = risky_outcome
        expected = np.full((n, 2), CONFIG.stable_outcome)
        expected[rows, risky_action] = (
            probability * CONFIG.bad_outcome
            + (1.0 - probability) * CONFIG.good_outcome
        )

        environment = copy.deepcopy(base)
        environment["Y_all"] = outcomes
        environment["Y"] = outcomes[rows, action].reshape(-1, 1).astype(np.float32)
        environment["mu"] = expected.astype(np.float32)
        environments.append(environment)
    return environments


def environments(data, severity):
    shift = data["shifts"][severity_label(severity)]
    if data["mode"] == "covariate":
        return resampled_environments(data["test_base"], shift["indices"])
    return drifted_environments(data["test_base"], shift["probabilities"])


def result_path(args, severity, method, replication):
    label = severity_label(severity)
    condition = (
        f"nonlinear_shift={severity:.1f}"
        if args.dgp_mode == "covariate"
        else f"nonlinear_drift={severity:.1f}"
    )
    prefix = "Shift" if args.dgp_mode == "covariate" else "Drift"
    directory = args.sandbox_dir / "results" / args.dgp_mode / condition / method
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}_{label}-exp_id-{replication}-min.pkl"


def run_replication(args, replication, method):
    data = joblib.load(
        args.sandbox_dir / "data" / args.dgp_mode / f"exp-{replication}.pkl"
    )
    args.robust_met = method
    args.model_dir = args.sandbox_dir / "models" / args.dgp_mode / method
    args.model_dir.mkdir(parents=True, exist_ok=True)

    set_seed(1 + replication)
    first_severity = args.severities[0]
    test = {
        "base": copy.deepcopy(data["test_base"]),
        "envs": environments(data, first_severity),
    }
    policy_data = basics.initialize_setting(
        torch.device(args.device), copy.deepcopy(data["train"]), test, args
    )
    learned_policy = policy_tools.policy_learning(
        pol_dict=policy_data,
        exp_id=replication,
        device=torch.device(args.device),
    )

    for severity in args.severities:
        policy_data["shift_env"] = environments(data, severity)
        evaluated = policy_tools.policy_evaluation_min(
            pol_dict=policy_data,
            opt_pol_dict=learned_policy,
            device=torch.device(args.device),
        )
        result = {f"{method}_Q_min": evaluated["Q_min"]}
        path = result_path(args, severity, method, replication)
        joblib.dump(result, path)
        print(f"Saved {path}: {result}", flush=True)


def build_parser():
    sandbox_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgp_mode", choices=("covariate", "concept"), required=True)
    parser.add_argument("--start_rep", type=int, default=0)
    parser.add_argument("--num_reps", type=int, default=3)
    parser.add_argument(
        "--methods", nargs="+", choices=("robust", "non-robust"),
        default=("robust", "non-robust")
    )
    parser.add_argument(
        "--severities", nargs="+", type=float, choices=SEVERITIES,
        default=list(SEVERITIES)
    )
    parser.add_argument("--sandbox_dir", type=Path, default=sandbox_dir)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--type", default="nonlinear")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--density_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--num_treat", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--amb_radius", type=float, default=0.2)
    parser.add_argument("--Prob_depth", type=int, default=2)
    parser.add_argument("--Prob_act_list", default=["ReLU", "Sigmoid"])
    parser.add_argument("--Prob_neurons_list", default=[16, 8])
    parser.add_argument("--Prob_dropout_list", default=[0, 0])
    parser.add_argument("--norm_met", default="normalized")
    parser.add_argument("--propensity_met", default="known")
    parser.add_argument("--opt_procedure", default="data-driven")
    parser.add_argument("--alpha_lb", type=float, default=0.1)
    parser.add_argument("--test_size", type=int, default=2500)
    parser.add_argument("--optimization_epochs", type=int, default=1000)
    parser.add_argument("--estimator_choices", default="normalized_estimator")
    parser.add_argument("--baseline", default="None")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot_boundary", action="store_true")
    parser.add_argument("--dist_shift", type=float, default=0.0)
    return parser


def main():
    args = build_parser().parse_args()
    args.sandbox_dir = args.sandbox_dir.resolve()
    args.mode = "shift" if args.dgp_mode == "covariate" else "drift"
    for replication in range(args.start_rep, args.start_rep + args.num_reps):
        for method in args.methods:
            print(
                f"Mode {args.dgp_mode}, replication {replication}, method {method}",
                flush=True,
            )
            run_replication(args, replication, method)


if __name__ == "__main__":
    main()
