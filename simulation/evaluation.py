"""Policy-value estimators used by Tables 1--3."""

import numpy as np
import torch


def dro_value(policy, x, action, outcome, propensity, log_alpha, args, radius=None):
    logits = policy(x)
    chosen = torch.max(torch.softmax(logits, dim=1), dim=1).indices.reshape(-1, 1)
    match = torch.where(chosen == action, 1, 0)
    if args.estimator_choices == "normalized_estimator":
        normalizer = match / propensity
    else:
        normalizer = torch.Tensor([1])

    alpha = torch.nn.functional.softplus(log_alpha)
    samples = match * torch.exp(-outcome / alpha) / propensity
    radius = args.amb_radius if radius is None else radius
    negative_value = (
        alpha * torch.log(torch.mean(samples) / torch.mean(normalizer))
        + alpha * radius
    )
    return -negative_value, logits


def ipw_value(policy, x, action, outcome, propensity, args):
    logits = policy(x)
    chosen = torch.max(torch.softmax(logits, dim=1), dim=1).indices.reshape(-1, 1)
    match = torch.where(chosen == action, 1, 0)
    if args.estimator_choices == "normalized_estimator":
        normalizer = match / propensity
    else:
        normalizer = torch.Tensor([1])
    return torch.mean(match * outcome / propensity) / torch.mean(normalizer)


def chosen_actions(policy, x):
    with torch.no_grad():
        return policy(x).argmax(1).cpu().numpy()


def worst_covariate_value(test, indices, actions):
    values = test["Y_all"][indices, actions[indices]]
    return float(values.mean(axis=1).min())


def concept_outcomes(test, probabilities, config):
    n = test["X"].shape[0]
    risky_action = test["risky_action"].reshape(-1)
    uniforms = test["catastrophe_uniform"].reshape(-1)
    risky = np.where(
        uniforms[None, :] < probabilities,
        config["bad_outcome"],
        config["good_outcome"],
    )
    outcomes = np.full((probabilities.shape[0], n, 2), config["stable_outcome"])
    outcomes[:, np.arange(n), risky_action] = risky
    return outcomes.astype(np.float32)


def worst_concept_value(test, probabilities, actions, config):
    outcomes = concept_outcomes(test, probabilities, config)
    rows = np.arange(test["X"].shape[0])
    values = outcomes[:, rows, actions]
    return float(values.mean(axis=1).min())
