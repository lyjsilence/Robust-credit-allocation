import opt_policy_learn_with_holdout as opt_policy_learn
import basics_with_holdout as basics
import numpy as np
import argparse
import random
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def severity_label(value):
    """Match the numeric labels used by the generated simulation files."""
    value = float(value)
    if value == 0:
        return "0"
    if value.is_integer():
        return str(int(value))
    return str(value)


def condition_name(args):
    """Return the directory name shared by data and saved models."""
    if args.mode == "raw":
        return args.type
    return f"nonlinear_{args.mode}={args.dist_shift:.1f}"


def resolve_data_dir(args):
    """Return the directory containing the requested DGP replications."""

    simulation_dir = Path(__file__).resolve().parent / "data"
    return simulation_dir / condition_name(args)


def model_directory(args):
    """Return the model directory for the requested DGP condition."""
    return Path("models/Simulation") / condition_name(args) / args.robust_met


def data_file_path(data_dir, mode, dist_shift, replication):
    if mode == "raw":
        return data_dir / f"Shift_0.0-exp-{replication}.pkl"
    prefix = "Shift" if mode == "shift" else "Drift"
    return data_dir / (
        f"{prefix}_{severity_label(dist_shift)}-exp-{replication}.pkl"
    )


'''This script focuses on data-driven procedure.'''
parser = argparse.ArgumentParser(description="Credit Allocation")
parser.add_argument("--type", type=str, default='nonlinear', choices=['linear', 'nonlinear'])

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--density_epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=0)

parser.add_argument("-num_treat", type=int, default=2)
parser.add_argument("-max_iter", type=int, default=100)
parser.add_argument("-amb_radius", type=float, default=0.2)

parser.add_argument("-Prob_depth", type=int, default=2)
parser.add_argument("-Prob_act_list", default=["ReLU", "Sigmoid"])
parser.add_argument("-Prob_neurons_list", default=[16, 8])
parser.add_argument("-Prob_dropout_list", default=[0, 0])

parser.add_argument("-norm_met", type=str, default="normalized", choices=["normalized", "unnormalized"])
parser.add_argument("-robust_met", type=str, default="robust", choices=["robust", "non-robust"])
parser.add_argument("-propensity_met", type=str, default="known", choices=["unknown", "known"])
parser.add_argument("-opt_procedure", type=str, default="data-driven", choices=["data-driven", "non-data-driven"])

parser.add_argument("-alpha_lb", type=float, default=0.1)
parser.add_argument(
    "--dist_shift",
    type=float,
    default=0,
    choices=[0, 0.5, 1, 1.5, 2, 2.5, 3],
)
parser.add_argument("--test_size", type=int, default=2500, choices=[500, 2000, 2500])

parser.add_argument("-optimization_epochs", type=int, default=1000)
parser.add_argument("-estimator_choices", type=str, default="normalized_estimator",
                    choices=["normalized_estimator", "unnormalized_estimator"])

parser.add_argument("-baseline", type=str, default='None', choices=['None', 'SNLN', 'KLLN'])
parser.add_argument("-verbose", type=bool, default=False, help='Print the training process or not')
parser.add_argument("-plot_boundary", type=bool, default=False, help='Plot the decision boundary in evaluation')
parser.add_argument(
    "--mode",
    type=str,
    default="raw",
    choices=["raw", "shift", "drift"],
)
parser.add_argument(
    "--start_rep",
    type=int,
    default=0,
    help="Index of the first replication to run.",
)
parser.add_argument(
    "--num_reps",
    type=int,
    default=100,
    help="Number of consecutive replications to run.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default='data/',
    help="Directory containing the requested simulation PKL files.",
)
parser.add_argument(
    "--output_root",
    type=str,
    default=".",
    help="Root directory for results/ and models/ (default: current directory).",
)

args = parser.parse_args()

if __name__ == '__main__':
    if args.start_rep < 0:
        parser.error("--start_rep must be nonnegative.")
    if args.num_reps < 1:
        parser.error("--num_reps must be at least 1.")
    if args.mode == "raw" and args.dist_shift != 0:
        parser.error("raw mode requires --dist_shift 0.")

    data_dir = resolve_data_dir(args)
    replications = range(args.start_rep, args.start_rep + args.num_reps)
    data_files = [
        data_file_path(data_dir, args.mode, args.dist_shift, i)
        for i in replications
    ]
    missing_files = [path for path in data_files if not path.is_file()]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        parser.error(f"Missing simulation data file(s):\n{missing}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(output_root)

    set_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''The policy list that can be run.'''
    # save the results
    if not os.path.exists(f"./results/Simulation/{args.type}/{args.robust_met}"):
        os.makedirs(f"./results/Simulation/{args.type}/{args.robust_met}")
    # save the models
    args.model_dir = model_directory(args)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline != 'None':
        args.opt_procedure = 'non-data-driven'

    for i, data_file in zip(replications, data_files):
        print(f'Training for dataset {i}')
        load_data = joblib.load(data_file)

        '''Policy Learning'''
        train_dict, test_dict = load_data["train"], load_data["test"]

        dict_train_forOPL = basics.initialize_setting(device, train_dict, test_dict, args)
        pol_dict_train = opt_policy_learn.policy_learning(pol_dict=dict_train_forOPL, exp_id=i, device=device)

        '''policy evaluation'''
        eval_dict = {}
        shift_label = (
            "0.0" if args.mode == "raw" else severity_label(args.dist_shift)
        )

        if args.mode == "raw":
            for delta in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
                dict_evaluation = dict_train_forOPL

                pol_dict_test = opt_policy_learn.policy_evaluation_dro(
                    pol_dict=dict_evaluation,
                    opt_pol_dict=pol_dict_train,
                    evaluator_type=args.robust_met,
                    amb_radius=delta,
                    device=device,
                )

                q_keys = [k for k in pol_dict_test if isinstance(k, str) and k.startswith('Q')]
                for k in q_keys:
                    eval_dict[args.robust_met + '_' + k + '_delta_' + str(delta)] = pol_dict_test[k]
                if args.baseline == 'None':
                    joblib.dump(eval_dict,
                                f"results/Simulation/{args.type}/{args.robust_met}/Shift_{shift_label}-exp_id-{i}-DRO.pkl")
                else:
                    joblib.dump(eval_dict,
                                f"results/Simulation/{args.type}/{args.robust_met}/Shift_{shift_label}-{args.baseline}-exp_id-{i}-DRO.pkl")
        else:
            dict_evaluation = dict_train_forOPL
            pol_dict_test = opt_policy_learn.policy_evaluation_min(
                pol_dict=dict_evaluation,
                opt_pol_dict=pol_dict_train,
                device=device,
            )

            eval_dict[args.robust_met + '_Q_min'] = pol_dict_test['Q_min']
            print(eval_dict)
            if args.mode == 'shift':
                joblib.dump(eval_dict,
                            f"results/Simulation/{args.type}/{args.robust_met}/Shift_{shift_label}-exp_id-{i}-min.pkl")
            elif args.mode == 'drift':
                joblib.dump(eval_dict,
                            f"results/Simulation/{args.type}/{args.robust_met}/Drift_{shift_label}-exp_id-{i}-min.pkl")
