"""Ordinary bootstrap OOB calibration on validation-only candidate predictions.

Widths must be increasing. Internal C multiplies sqrt(k / n_total); the
reported validation-scale c is C * sqrt(n_validation / n_total). Maximizers
within 1e-10 are tied, with the smaller width or coefficient selected first.
No test observations or scores enter these functions.
"""
import numpy as np

from policy_core.robust_values import robust_values

TOL = 1e-10


def coefficient_grid():
    """Frozen 413-point grid in internal C units for the N=5000 experiment."""
    anchors = np.r_[0., .008*np.sqrt(10), .08*np.sqrt(10),
                    [.005, .01, .02, .04, .08, .16, .32, .64, 1.28]]
    values = list(anchors)
    for c in np.geomspace(.0001, 4., 401)*np.sqrt(10):
        if not any(abs(c-x) <= 1e-14*max(1., abs(x)) for x in values):
            values.append(float(c))
    return np.sort(values)


def bootstrap_counts(n, seed, draws=200):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n)
                     for _ in range(draws)])


def choices(values, widths, coefficients, n=5000):
    """Smallest-width maximizer within TOL, for each sample and coefficient."""
    values = np.asarray(values, float)
    one = values.ndim == 1
    if one:
        values = values[None, :]
    if not np.isfinite(values).all():
        raise ValueError('Invalid values cannot enter calibration')
    h = np.sqrt(np.asarray(widths)/n)
    pieces = []
    for start in range(0, len(coefficients), 128):
        score = values[:, None, :] - np.asarray(coefficients)[None, start:start+128, None]*h
        pieces.append(np.argmax(score >= score.max(-1, keepdims=True)-TOL, axis=-1))
    result = np.concatenate(pieces, axis=1)
    return result[0] if one else result


def argbest(scores, coefficients):
    scores = np.asarray(scores)
    return float(np.asarray(coefficients)[np.flatnonzero(scores >= scores.max()-TOL)[0]])


def oob_scores(inside, outside, widths, coefficients, n=5000):
    indices = choices(inside, widths, coefficients, n)
    return np.mean(np.take_along_axis(outside, indices, axis=1), axis=0)


def validation_scores(data, decisions, counts, eta=.2):
    """Reoptimize the KL dual on full, in-bag and OOB validation samples."""
    kw = dict(y=data['Y'], actions=data['T'], decisions=decisions,
              propensity=data['prop_score'], eta=eta, minimum_matches=5)
    full = robust_values(**kw)['value'][0]
    inside = robust_values(**kw, counts=counts)
    outside = robust_values(**kw, counts=(counts == 0).astype(int))
    valid = inside['valid'].all(1) & outside['valid'].all(1)
    if valid.mean() < .95:
        raise ValueError('More than 5% of bootstrap draws have insufficient matches')
    return dict(full=full, inside=inside['value'], outside=outside['value'], valid=valid)


def select_oob(full, inside, outside, valid, widths, coefficients, cap,
               n_total=5000, n_validation=500):
    """Calibrate C within cap, then select a policy on the full validation set.

    The valid-draw mask uses the whole candidate pool, including when a smaller
    cap is displayed, exactly as in the original 100-replication experiment.
    """
    widths = np.asarray(widths)
    if np.any(np.diff(widths) <= 0) or not np.any(widths <= cap):
        raise ValueError('Widths must be increasing and the cap must include a candidate')
    if np.mean(valid) < .95:
        raise ValueError('At least 95% valid bootstrap draws are required')
    keep = widths <= cap
    w = widths[keep]
    curve = oob_scores(inside[valid][:, keep], outside[valid][:, keep], w, coefficients, n_total)
    coefficient = argbest(curve, coefficients)
    index = int(choices(full[keep], w, [coefficient], n_total)[0])
    return dict(width=int(w[index]), coefficient=coefficient,
                old_c=coefficient/np.sqrt(n_total/n_validation),
                validation_value=float(full[keep][index])), curve


def calibrate_oob(data, decisions, widths, seed, cap, *, n_total=5000,
                  eta=.2, draws=200, coefficients=None):
    """Public entry point after fitting candidates: returns selection and scores.

    Supply held-out validation data/predictions only; train/test data are not
    accepted. Pass an explicit grid when using settings other than N=5000.
    """
    coefficients = coefficient_grid() if coefficients is None else np.asarray(coefficients)
    counts = bootstrap_counts(len(data['Y']), seed, draws)
    scores = validation_scores(data, decisions, counts, eta)
    selection, curve = select_oob(**scores, widths=widths, coefficients=coefficients,
                                 cap=cap, n_total=n_total, n_validation=len(data['Y']))
    return selection, dict(**scores, counts=counts, coefficients=coefficients, curve=curve)
