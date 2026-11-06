"""Generate observations for the complexity and perturbation experiments."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from .shift import SEVERITIES, nominal_outcomes

DESIGN = json.loads(Path(__file__).with_name('design.json').read_text(encoding='utf-8'))
with np.load(Path(__file__).with_name('random_seeds.npz')) as archive:
    SEEDS = {name:archive[name] for name in archive.files}
CASES = ('low', 'moderate', 'high')
SPLITS = ('master_train', 'master_validation', 'test')


def shift_direction_seeds(rep):
    return SEEDS['shift_directions'][rep].tolist()


def shift_resampling_seed(rep, severity, environment):
    index = SEVERITIES[1:].index(severity)
    return int(SEEDS['shift_resampling'][rep, index, environment])


def fitting_seeds(study, case, rep, width):
    if width not in range(2, 257, 2):
        raise ValueError('Width must be an even integer from 2 through 256')
    if study == 'shift':
        row = SEEDS['shift_fitting'][rep, width//2-1]
    else:
        row = SEEDS['baseline_fitting'][rep, CASES.index(case), width//2-1]
    return tuple(int(value) for value in row)


def teacher(frequency):
    """Exact ReLU representation of the two-coordinate triangle score."""
    f = frequency
    net = nn.Sequential(nn.Linear(4, 6*f), nn.ReLU(), nn.Linear(6*f, 1)).double()
    with torch.no_grad():
        net[0].weight.zero_(); net[0].bias.zero_()
        net[2].weight.zero_(); net[2].bias.fill_(-1)
        for axis in range(2):
            for j in range(f):
                for k, (offset, coef) in enumerate(((0, 2*f), (.5, -4*f), (1, 2*f))):
                    i = axis*3*f + 3*j + k
                    net[0].weight[i, axis] = .5
                    net[0].bias[i] = .5-(j+offset)/f
                    net[2].weight[0, i] = coef
    net.requires_grad_(False)
    return net.eval()


def smooth(case, n, seed):
    norm = DESIGN['normalization'][case]
    net = teacher(CASES.index(case)+1)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 4))
    with torch.no_grad():
        score = (net(torch.from_numpy(x)).numpy().ravel()-norm['median'])/norm['sd']
    delta = 1.5*np.tanh(2*score)
    outcomes = expit(np.column_stack((-delta, delta))+rng.normal(0, .6, (n, 1)))
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    offsets = np.sqrt(2)*.6*nodes
    means = np.column_stack([expit(sign*delta[:, None]+offsets)@weights/np.sqrt(np.pi)
                            for sign in (-1, 1)])
    propensity = np.full((n, 2), .5)
    actions = np.asarray([rng.choice(2, p=row) for row in propensity], dtype=np.int64)
    rows = np.arange(n)
    return dict(X=x.astype(np.float32), T=actions.reshape(-1, 1),
        Y=outcomes[rows, actions].reshape(-1, 1).astype(np.float32),
        prop_score=propensity[rows, actions].reshape(-1, 1).astype(np.float32),
        Y_all=outcomes.astype(np.float32), prop_score_all=propensity.astype(np.float32),
        mu=means.astype(np.float32))


def training_data(study, case, rep):
    if study == 'shift' and case != 'moderate':
        raise ValueError('The perturbation experiment uses the Middle case only')
    seeds = SEEDS['baseline_data'][rep, CASES.index(case)]
    result = {split:smooth(case, DESIGN['sample_sizes'][split], int(seeds[i]))
              for i, split in enumerate(SPLITS)}
    if study == 'shift':
        for i, split in enumerate(SPLITS):
            part = result[split]
            rng = np.random.default_rng(int(SEEDS['shift_outcomes'][rep, i]))
            n = len(part['X'])
            g = nominal_outcomes(part['X'], part['mu'].argmax(1),
                                 rng.uniform(size=n), rng.normal(0, .6, size=n))
            part['Y_all'] = g['Y_all'].astype(np.float32)
            part['Y'] = g['Y_all'][np.arange(n), part['T'].ravel(), None].astype(np.float32)
            part['mu'] = g['mu'].astype(np.float32)
            for key, source in (('tail_p0', 'p0'), ('tail_u', 'U'),
                                ('tail_z', 'Z'), ('tail_risky_action', 'risky_action')):
                part[key] = g[source].reshape(-1, 1).astype(
                    np.int64 if key == 'tail_risky_action' else np.float64)
    return result


def held_out_data(case, rep, split):
    index = ('validation', 'test').index(split)
    seed = int(SEEDS['baseline_evaluation'][rep, CASES.index(case), index])
    size = DESIGN['sample_sizes']['master_validation' if split == 'validation' else 'test']
    return smooth(case, size, seed)


def oob_counts(study, case, rep):
    from policy_core.oob import bootstrap_counts
    if study == 'shift':
        return bootstrap_counts(500, int(SEEDS['shift_oob'][rep]))
    if rep < 20:
        # Original resampling draws are fixed design inputs, not observations.
        with np.load(Path(__file__).with_name('bootstrap_counts.npz')) as z:
            return z[f'{case}_{rep}'].astype(np.int64)
    return bootstrap_counts(500, int(SEEDS['baseline_oob'][rep-20, CASES.index(case)]))
