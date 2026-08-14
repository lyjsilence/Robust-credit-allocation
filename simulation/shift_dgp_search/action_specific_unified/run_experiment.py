"""Run Tables 1--3 and boundary figures in the unified DGP sandbox."""

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


HERE = Path(__file__).resolve().parent
SIMULATION_DIR = HERE.parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

import alternative_optimization_with_holdout as evaluation_tools
import basics_with_holdout as basics
import opt_policy_learn_with_holdout as policy_tools
import optimization_problem_with_holdout as optimization
from dgp import SEVERITIES, concept_outcomes, generate_replication


RADII = (0.05, 0.10, 0.20, 0.30, 0.40)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_args(cli, model_dir, robust_met, baseline="None"):
    args = copy.copy(cli)
    args.mode = "raw"
    args.type = "nonlinear"
    args.dist_shift = 0.0
    args.robust_met = robust_met
    args.baseline = baseline
    args.opt_procedure = "data-driven" if baseline == "None" else "non-data-driven"
    args.model_dir = model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    return args


def fit_candidates(cli, train, test, replication, name, depths, robust_met, baseline="None"):
    model_dir = cli.out_dir / "models" / f"rep-{replication}" / name
    args = make_args(cli, model_dir, robust_met, baseline)
    seed = 100000 * replication + sum(ord(char) for char in name)
    set_seed(seed)
    policy_data = basics.initialize_setting(
        torch.device(cli.device), copy.deepcopy(train), copy.deepcopy(test), args
    )

    candidates = {}
    for depth in depths:
        policy_data[f"alpha_list_{depth}"] = (
            None if robust_met == "non-robust" else [policy_data["init_alpha"]]
        )
        path, value, alphas = policy_tools.pi_alpha_alter(
            policy_data, cli.num_treat, depth, replication, torch.device(cli.device)
        )
        candidates[depth] = {
            "path": path,
            "value": value,
            "criterion": value - np.sqrt(depth / policy_data["X_val"].shape[0]),
            "alpha": alphas[-1] if alphas else None,
        }
    selected = max(candidates, key=lambda depth: candidates[depth]["criterion"])
    return policy_data, candidates, selected


def load_policy(policy_data, candidate, depth, baseline, device):
    if baseline == "None":
        policy = optimization.NN_Sieves(
            policy_data, policy_data["dim_X"], device, Nlayers=depth,
            Nclasses=policy_data["num_treat"]
        )
    else:
        policy = optimization.NN_baseline(
            policy_data, policy_data["dim_X"], device, Nlayers=depth,
            Nclasses=policy_data["num_treat"]
        )
    policy.load_state_dict(torch.load(candidate["path"], map_location=device))
    policy.eval()
    optimization.set_requires_grad(policy, False)
    return policy


def robust_value(policy_data, candidate, depth, baseline, radius, device):
    policy = load_policy(policy_data, candidate, depth, baseline, device)
    log_alpha = torch.tensor([candidate["alpha"]], device=device)
    value, _ = evaluation_tools.evaluation(
        policy,
        torch.tensor(policy_data["X_test"], device=device),
        torch.tensor(policy_data["T_test"], device=device),
        torch.tensor(policy_data["Y_test"], device=device),
        torch.tensor(policy_data["prop_score_test"], device=device),
        log_alpha,
        policy_data["args"],
        amb_radius=radius,
    )
    return float(value.item())


def chosen_actions(policy_data, candidate, depth, device):
    policy = load_policy(policy_data, candidate, depth, "None", device)
    with torch.no_grad():
        logits = policy(torch.tensor(policy_data["X_test"], device=device))
    return logits.argmax(1).cpu().numpy()


def worst_covariate_value(test, indices, actions):
    values = test["Y_all"][indices, actions[indices]]
    return float(values.mean(axis=1).min())


def worst_concept_value(test, probabilities, actions):
    outcomes = concept_outcomes(test, probabilities)
    rows = np.arange(test["X"].shape[0])
    values = outcomes[:, rows, actions]
    return float(values.mean(axis=1).min())


def run_replication(cli, replication):
    device = torch.device(cli.device)
    data_path = cli.out_dir / "data" / f"rep-{replication}.pkl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        data = joblib.load(data_path)
    else:
        data = generate_replication(
            replication, cli.n_train, cli.n_test, cli.num_environments
        )
        joblib.dump(data, data_path, compress=3)

    result = {"replication": replication, "table1": [], "table2": [], "table3": []}

    nonlinear = data["nonlinear"]
    nl_policy, nl_candidates, nl_selected = fit_candidates(
        cli, nonlinear["train"], nonlinear["test"], replication,
        "nonlinear_ours_robust", range(1, 7), "robust"
    )
    result["selected_depth"] = nl_selected

    for depth, candidate in nl_candidates.items():
        for radius in RADII:
            value = robust_value(nl_policy, candidate, depth, "None", radius, device)
            result["table3"].append(
                {"complexity": f"K={depth}", "radius": radius, "value": value}
            )
    for radius in RADII:
        value = robust_value(
            nl_policy, nl_candidates[nl_selected], nl_selected, "None", radius, device
        )
        result["table1"].append(
            {"boundary": "nonlinear", "model": "Ours", "radius": radius, "value": value}
        )
        result["table3"].append(
            {"complexity": "Data-driven", "radius": radius, "value": value}
        )

    tail = data["tail"]
    tail_robust, tail_r_candidates, tail_r_selected = fit_candidates(
        cli, tail["train"], tail["test"], replication,
        "tail_ours_robust", (1, 2), "robust"
    )
    tail_nonrobust, tail_nr_candidates, tail_nr_selected = fit_candidates(
        cli, tail["train"], tail["test"], replication,
        "tail_ours_nonrobust", (1, 2), "non-robust"
    )
    robust_actions = chosen_actions(
        tail_robust, tail_r_candidates[tail_r_selected], tail_r_selected, device
    )
    nonrobust_actions = chosen_actions(
        tail_nonrobust, tail_nr_candidates[tail_nr_selected], tail_nr_selected, device
    )
    for mode in ("covariate", "concept"):
        for severity in SEVERITIES:
            specification = data[mode][f"{severity:g}"]
            if mode == "covariate":
                robust = worst_covariate_value(
                    tail["test"], specification["indices"], robust_actions
                )
                nonrobust = worst_covariate_value(
                    tail["test"], specification["indices"], nonrobust_actions
                )
            else:
                robust = worst_concept_value(
                    tail["test"], specification["probabilities"], robust_actions
                )
                nonrobust = worst_concept_value(
                    tail["test"], specification["probabilities"], nonrobust_actions
                )
            result["table2"].extend(
                [
                    {"mode": mode, "severity": severity, "method": "Robust", "value": robust},
                    {"mode": mode, "severity": severity, "method": "Non-robust", "value": nonrobust},
                ]
            )

    for boundary in ("linear", "nonlinear"):
        sample = data[boundary]
        snls_policy, snls_candidates, _ = fit_candidates(
            cli, sample["train"], sample["test"], replication,
            f"{boundary}_snls", (2,), "robust", baseline="SNLN"
        )
        for radius in RADII:
            value = robust_value(
                snls_policy, snls_candidates[2], 2, "SNLN", radius, device
            )
            result["table1"].append(
                {"boundary": boundary, "model": "SNLS", "radius": radius, "value": value}
            )

    linear = data["linear"]
    linear_policy, linear_candidates, linear_selected = fit_candidates(
        cli, linear["train"], linear["test"], replication,
        "linear_ours_robust", (1, 2), "robust"
    )
    result["linear_selected_depth"] = linear_selected
    for radius in RADII:
        value = robust_value(
            linear_policy, linear_candidates[linear_selected], linear_selected,
            "None", radius, device
        )
        result["table1"].append(
            {"boundary": "linear", "model": "Ours", "radius": radius, "value": value}
        )

    result_path = cli.out_dir / "results" / f"rep-{replication}.pkl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, result_path)
    print(f"Completed replication {replication}: selected K={nl_selected}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_rep", type=int, default=0)
    parser.add_argument("--num_reps", type=int, default=3)
    parser.add_argument("--n_train", type=int, default=10000)
    parser.add_argument("--n_test", type=int, default=2500)
    parser.add_argument("--num_environments", type=int, default=100)
    parser.add_argument("--out_dir", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--density_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_treat", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--amb_radius", type=float, default=0.20)
    parser.add_argument("--Prob_depth", type=int, default=2)
    parser.add_argument("--Prob_act_list", default=["ReLU", "Sigmoid"])
    parser.add_argument("--Prob_neurons_list", default=[16, 8])
    parser.add_argument("--Prob_dropout_list", default=[0, 0])
    parser.add_argument("--norm_met", default="normalized")
    parser.add_argument("--propensity_met", default="known")
    parser.add_argument("--alpha_lb", type=float, default=0.1)
    parser.add_argument("--test_size", type=int, default=2500)
    parser.add_argument("--optimization_epochs", type=int, default=1000)
    parser.add_argument("--estimator_choices", default="normalized_estimator")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot_boundary", action="store_true")
    return parser


def main():
    cli = build_parser().parse_args()
    cli.out_dir = cli.out_dir.resolve()
    for replication in range(cli.start_rep, cli.start_rep + cli.num_reps):
        run_replication(cli, replication)


if __name__ == "__main__":
    main()
