"""Policy values and ordinary bootstrap out-of-bag width selection."""

import numpy as np

import torch

TOLERANCE = 1e-10


def coefficient_grid():
    """413 penalty coefficients multiplying sqrt(width / 5000)."""
    anchors = np.r_[
        0.0,
        0.008 * np.sqrt(10),
        0.08 * np.sqrt(10),
        [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28],
    ]
    values = list(anchors)
    for c in np.geomspace(0.0001, 4.0, 401) * np.sqrt(10):
        if not any(abs(c - x) <= 1e-14 * max(1.0, abs(x)) for x in values):
            values.append(float(c))
    return np.sort(values)


def bootstrap_counts(n, seed, draws=200):
    rng = np.random.default_rng(seed)
    return np.stack(
        [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]
    )


def choices(values, widths, coefficients, n=5000):
    """Smallest-width maximizer within tolerance, for each sample and coefficient."""
    values = np.asarray(values, float)
    single_sample = values.ndim == 1
    values = np.atleast_2d(values)
    if not np.isfinite(values).all():
        raise ValueError("Invalid values cannot enter calibration")
    penalty = np.sqrt(np.asarray(widths) / n)
    pieces = []
    for start in range(0, len(coefficients), 128):
        score = (
            values[:, None, :]
            - np.asarray(coefficients)[None, start : start + 128, None] * penalty
        )
        pieces.append(
            np.argmax(score >= score.max(-1, keepdims=True) - TOLERANCE, axis=-1)
        )
    result = np.concatenate(pieces, axis=1)
    return result[0] if single_sample else result


def validation_scores(data, decisions, counts, eta=0.2):
    """Reoptimize the KL dual on full, in-bag and OOB validation samples."""
    arguments = dict(
        y=data["Y"],
        actions=data["T"],
        decisions=decisions,
        propensity=data["prop_score"],
        eta=eta,
        minimum_matches=5,
    )
    full = robust_values(**arguments)["value"][0]
    inside = robust_values(**arguments, counts=counts)
    outside = robust_values(**arguments, counts=(counts == 0).astype(int))
    valid = inside["valid"].all(1) & outside["valid"].all(1)
    if valid.mean() < 0.95:
        raise ValueError("More than 5% of bootstrap draws have insufficient matches")
    return dict(
        full=full, inside=inside["value"], outside=outside["value"], valid=valid
    )


def select_oob(
    full,
    inside,
    outside,
    valid,
    widths,
    coefficients,
    cap,
    n_total=5000,
    n_validation=500,
):
    """Choose the OOB coefficient, then the full-validation width.

    Widths and coefficients are increasing. The valid mask covers the entire
    candidate pool, including widths above the reporting cap.
    """
    keep = np.asarray(widths) <= cap
    widths = np.asarray(widths)[keep]
    indices = choices(inside[valid][:, keep], widths, coefficients, n_total)
    curve = np.mean(
        np.take_along_axis(outside[valid][:, keep], indices, axis=1), axis=0
    )
    coefficient = float(
        coefficients[np.flatnonzero(curve >= curve.max() - TOLERANCE)[0]]
    )
    index = int(choices(full[keep], widths, [coefficient], n_total)[0])
    return (
        dict(
            width=int(widths[index]),
            coefficient=coefficient,
            validation_coefficient=coefficient / np.sqrt(n_total / n_validation),
            validation_value=float(full[keep][index]),
        ),
        curve,
    )


def robust_values(
    y,
    actions,
    decisions,
    propensity,
    eta,
    counts=None,
    device="cpu",
    batch_size=8,
    minimum_matches=1,
):
    """Return B x K values and validity masks for paired resampling counts.

    Solve KL(exponentially tilted empirical weights || empirical weights)=eta
    by bisection in 1/alpha. Shift by each weighted minimum for stability; this
    changes no estimator. Include the alpha->0 endpoint when the empirical
    ambiguity ball reaches the minimum. No artificial positive lower/upper cap.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    a = np.asarray(actions).ravel()
    decisions = np.asarray(decisions)
    p = np.asarray(propensity, dtype=np.float64).ravel()
    if counts is None:
        counts = np.ones((1, len(y)))
    counts = np.asarray(counts, dtype=np.float64)
    dtype = torch.float64
    yt = torch.as_tensor(y, dtype=dtype, device=device)[None, None, :]
    base = torch.as_tensor(
        (decisions.T == a[None, :]) / p[None, :], dtype=dtype, device=device
    )[None, :, :]
    collected = {k: [] for k in ("value", "valid")}
    with torch.no_grad():
        for start in range(0, len(counts), batch_size):
            ct = torch.as_tensor(
                counts[start : start + batch_size], dtype=dtype, device=device
            )[:, None, :]
            w = ct * base
            total = w.sum(-1)
            valid = total > 0
            unique = (w > 0).sum(-1)
            valid = valid & (unique >= minimum_matches)
            norm = w / total.clamp_min(1e-300)[:, :, None]
            ymin = torch.where(w > 0, yt, torch.inf).min(-1).values
            ymin = torch.where(total > 0, ymin, torch.zeros_like(ymin))
            z = yt - ymin[:, :, None]
            z = torch.where(w > 0, z, torch.zeros_like(z))
            minmass = (norm * (z == 0)).sum(-1)
            limit_kl = -torch.log(minmass.clamp_min(1e-300))
            endpoint = limit_kl <= eta + 1e-13

            def tilted(t):
                terms = norm * torch.exp(-t[:, :, None] * z)
                partition = terms.sum(-1).clamp_min(1e-300)
                mean_z = (terms * z).sum(-1) / partition
                kl = -t * mean_z - torch.log(partition)
                return kl, partition

            lo = torch.zeros_like(total)
            hi = torch.ones_like(total)
            if eta == 0:
                value = (norm * yt).sum(-1)
            else:
                for _ in range(64):
                    kl, _ = tilted(hi)
                    expand = (kl < eta) & valid & ~endpoint
                    if not bool(expand.any()):
                        break
                    hi = torch.where(expand, hi * 2, hi)
                else:
                    raise RuntimeError("Could not bracket positive dual optimum")
                for _ in range(42):
                    mid = (lo + hi) / 2
                    kl, _ = tilted(mid)
                    lo = torch.where(kl < eta, mid, lo)
                    hi = torch.where(kl >= eta, mid, hi)
                t = (lo + hi) / 2
                kl, partition = tilted(t)
                value = torch.where(
                    endpoint, ymin, ymin - (torch.log(partition) + eta) / t
                )
            values = dict(value=torch.where(valid, value, torch.nan), valid=valid)
            for key, tensor in values.items():
                collected[key].append(tensor.cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}
