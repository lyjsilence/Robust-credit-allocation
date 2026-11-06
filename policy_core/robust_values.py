"""Batched dual optimization of self-normalized KL policy values."""
import numpy as np
import torch


def robust_values(y, actions, decisions, propensity, eta, counts=None,
                  device='cpu', batch_size=8, minimum_matches=1):
    """Return B x K values/alpha/diagnostics for paired resampling counts.

    Solve KL(exponentially tilted empirical weights || empirical weights)=eta
    by bisection in 1/alpha. Shift by each weighted minimum for stability; this
    changes no estimator. Include the alpha->0 endpoint when the empirical
    ambiguity ball reaches the minimum. No artificial positive lower/upper cap.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    a = np.asarray(actions).ravel()
    decisions = np.asarray(decisions)
    p = np.asarray(propensity, dtype=np.float64).ravel()
    if decisions.ndim != 2 or decisions.shape[0] != len(y):
        raise ValueError('Decisions must have shape observations x candidates')
    if not np.all(np.isfinite(y)) or not np.all((p > 0) & (p <= 1)) or eta < 0:
        raise ValueError('Invalid evaluation data/radius')
    if counts is None:
        counts = np.ones((1, len(y)))
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != len(y) or (counts < 0).any():
        raise ValueError('Invalid paired resampling counts')
    dtype = torch.float64
    yt = torch.as_tensor(y, dtype=dtype, device=device)[None, None, :]
    base = torch.as_tensor((decisions.T == a[None, :]) / p[None, :],
                           dtype=dtype, device=device)[None, :, :]
    collected = {k: [] for k in ('value', 'alpha', 'ess', 'matched', 'unique_matched', 'valid', 'kl_residual')}
    with torch.no_grad():
        for start in range(0, len(counts), batch_size):
            ct = torch.as_tensor(counts[start:start + batch_size], dtype=dtype, device=device)[:, None, :]
            w = ct * base
            total = w.sum(-1)
            valid = total > 0
            unique = (w > 0).sum(-1)
            matched = (ct * (base > 0)).sum(-1)
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
                alpha = torch.full_like(total, torch.inf)
                residual = torch.zeros_like(total)
            else:
                for _ in range(64):
                    kl, _ = tilted(hi)
                    expand = (kl < eta) & valid & ~endpoint
                    if not bool(expand.any()):
                        break
                    hi = torch.where(expand, hi * 2, hi)
                else:
                    raise RuntimeError('Could not bracket positive dual optimum')
                for _ in range(42):
                    mid = (lo + hi) / 2
                    kl, _ = tilted(mid)
                    lo = torch.where(kl < eta, mid, lo)
                    hi = torch.where(kl >= eta, mid, hi)
                t = (lo + hi) / 2
                kl, partition = tilted(t)
                alpha = torch.where(endpoint, torch.zeros_like(t), 1 / t)
                value = torch.where(endpoint, ymin, ymin - (torch.log(partition) + eta) / t)
                residual = torch.where(endpoint, torch.zeros_like(kl), torch.abs(kl - eta))
            ess = total.square() / w.square().sum(-1).clamp_min(1e-300)
            values = dict(value=torch.where(valid, value, torch.nan),
                          alpha=torch.where(valid, alpha, torch.nan), ess=ess,
                          matched=matched, unique_matched=unique, valid=valid,
                          kl_residual=torch.where(valid, residual, torch.nan))
            for key, tensor in values.items():
                collected[key].append(tensor.cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}
