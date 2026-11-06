"""Prepare train, validation, and test arrays for policy learning."""

import numpy as np


def prepare_policy_data(
    train,
    test,
    args,
    limit_test=None,
    alpha_bound_order="multiply",
    validation_fraction=0.2,
):
    validation_size = int(validation_fraction * train["X"].shape[0])
    validation = {key: value[:validation_size, :] for key, value in train.items()}
    train = {key: value[validation_size:, :] for key, value in train.items()}
    holdout_fraction = float(getattr(args, "optimization_holdout_fraction", 0.0))
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("optimization_holdout_fraction must lie in [0, 1)")
    holdout_size = int(holdout_fraction * train["X"].shape[0])
    if holdout_size:
        checkpoint = {key: value[:holdout_size, :] for key, value in train.items()}
        train = {key: value[holdout_size:, :] for key, value in train.items()}
    else:
        checkpoint = train
    if limit_test is not None:
        test = {key: value[:limit_test, :] for key, value in test.items()}

    policy_data = {
        "X": train["X"],
        "T": train["T"],
        "Y": train["Y"],
        "X_val": validation["X"],
        "T_val": validation["T"],
        "Y_val": validation["Y"],
        "X_test": test["X"],
        "T_test": test["T"],
        "Y_test": test["Y"],
        "prop_score": train["prop_score"],
        "prop_score_val": validation["prop_score"],
        "prop_score_test": test["prop_score"],
        "X_checkpoint": checkpoint["X"],
        "T_checkpoint": checkpoint["T"],
        "Y_checkpoint": checkpoint["Y"],
        "prop_score_checkpoint": checkpoint["prop_score"],
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
    if alpha_bound_order == "divide":
        policy_data["alpha_ub"] = policy_data["M"] / args.amb_radius
    else:
        policy_data["alpha_ub"] = 1 / args.amb_radius * policy_data["M"]
    policy_data["alpha_lb"] = args.alpha_lb
    policy_data["init_alpha"] = np.random.uniform(
        policy_data["alpha_lb"], policy_data["alpha_ub"]
    )

    for key in ("Y_all",):
        if key in train:
            policy_data[key] = train[key]
            policy_data[f"{key}_val"] = validation[key]
            policy_data[f"{key}_test"] = test[key]
            policy_data[f"{key}_checkpoint"] = checkpoint[key]
    return policy_data
