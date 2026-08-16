"""Prepare the arrays used by policy learning."""

import numpy as np


def prepare_policy_data(train, test, args):
    """Use the first 20% of training observations as the validation sample."""
    validation_size = int(0.2 * train["X"].shape[0])
    validation = {}
    for key in train:
        validation[key] = train[key][:validation_size, :]
        train[key] = train[key][validation_size:, :]

    test = {key: value[:args.test_size, :] for key, value in test.items()}
    policy_data = {
        "X": train["X"],
        "T": train["T"],
        "Y": train["Y"],
        "Y_all": train["Y_all"],
        "X_val": validation["X"],
        "T_val": validation["T"],
        "Y_val": validation["Y"],
        "Y_all_val": validation["Y_all"],
        "X_test": test["X"],
        "T_test": test["T"],
        "Y_test": test["Y"],
        "Y_all_test": test["Y_all"],
        "prop_score": train["prop_score"],
        "prop_score_val": validation["prop_score"],
        "prop_score_test": test["prop_score"],
        "tr_propensity_score_path": None,
        "propensity_met": args.propensity_met,
        "weight_decay": args.weight_decay,
        "amb_radius": args.amb_radius,
        "norm_met": args.norm_met,
        "robust_met": args.robust_met,
        "opt_procedure": args.opt_procedure,
        "estimator_choices": args.estimator_choices,
        "num_treat": args.num_treat,
        "dist_shift": args.dist_shift,
        "args": args,
        "dim_T": train["T"].shape[1],
        "dim_X": train["X"].shape[1],
    }
    policy_data["dim_TX"] = policy_data["dim_T"] + policy_data["dim_X"]
    policy_data["M"] = np.max((np.max(train["Y"]), np.max(test["Y"])))
    policy_data["alpha_ub"] = policy_data["M"] / args.amb_radius
    policy_data["alpha_lb"] = args.alpha_lb
    policy_data["init_alpha"] = np.random.uniform(
        policy_data["alpha_lb"], policy_data["alpha_ub"]
    )
    return policy_data
