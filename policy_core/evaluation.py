"""DRO and IPW policy-value estimators."""

import torch
import math
from torch.nn import functional as F


def soft_match(logits, action, temperature=1.0, eps=None):
    probabilities = torch.softmax(logits / temperature, dim=1)
    indices = action.squeeze().long()
    weights = probabilities[
        torch.arange(probabilities.size(0), device=probabilities.device), indices
    ].reshape(-1, 1)
    return torch.clamp(weights, min=eps) if eps is not None else weights


def hard_match(logits, action):
    chosen = torch.max(torch.softmax(logits, dim=1), dim=1).indices.reshape(-1, 1)
    return torch.where(chosen == action, 1, 0)


def dro_value(
    policy,
    x,
    action,
    outcome,
    propensity,
    log_alpha,
    args,
    matching="hard",
    temperature=1.0,
    radius=None,
):
    logits = policy(x.to(torch.float32))
    if matching == "soft":
        match = soft_match(logits, action, temperature, eps=1e-8)
        propensity = torch.clamp(propensity, min=1e-8)
    else:
        match = hard_match(logits, action)

    if args.estimator_choices == "normalized_estimator":
        normalizer = match / propensity
    else:
        # Hard matches are integer tensors; the unnormalized mean needs floats.
        normalizer = torch.ones_like(match, dtype=outcome.dtype)

    alpha = F.softplus(log_alpha)
    if matching == "soft":
        alpha = torch.clamp(alpha, min=1e-8)
    if (getattr(args, "alpha_upper_bound", None) is not None
            and args.estimator_choices == "unnormalized_estimator"):
        terms = torch.log(match.to(outcome.dtype)) - outcome / alpha - torch.log(propensity)
        log_ratio = torch.logsumexp(terms.reshape(-1), 0) - math.log(terms.numel())
        if matching == "soft":
            # Preserve the existing soft checkpoint estimator's ratio floor.
            log_ratio = torch.clamp(log_ratio, min=math.log(1e-8))
        radius = args.amb_radius if radius is None else radius
        return -alpha * (log_ratio + radius), logits
    samples = match * torch.exp(-outcome / alpha) / propensity
    ratio = torch.mean(samples) / torch.mean(normalizer)
    if matching == "soft":
        ratio = torch.clamp(ratio, min=1e-8)
    radius = args.amb_radius if radius is None else radius
    negative_value = alpha * torch.log(ratio) + alpha * radius
    return -negative_value, logits


def ipw_value(
    policy,
    x,
    action,
    outcome,
    propensity,
    args,
    matching="hard",
    temperature=1.0,
):
    logits = policy(x.to(torch.float32))
    if matching == "soft":
        match = soft_match(logits, action, temperature, eps=1e-8)
        propensity = torch.clamp(propensity, min=1e-8)
    else:
        match = hard_match(logits, action).float()
        propensity = torch.clamp(propensity, min=1e-8)

    if args.estimator_choices == "normalized_estimator":
        normalizer = match / propensity
    else:
        normalizer = torch.ones_like(match)
    denominator = torch.clamp(torch.mean(normalizer), min=1e-8)
    return torch.mean(match * outcome / propensity) / denominator
