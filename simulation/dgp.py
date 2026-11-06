"""Generate the two simulation designs from their fixed random streams."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from .shift import nominal_outcomes

DESIGN = json.loads(Path(__file__).with_name('design.json').read_text(encoding='utf-8'))
CASES = ('low', 'moderate', 'high')
SPLITS = ('master_train', 'master_validation', 'test')


def stream_seed(cycle, phase, batch, scenario, replication, role, width=0, attempt=0):
    identity = [str(cycle), str(phase), str(batch), str(scenario),
                int(replication), str(role), int(width), int(attempt)]
    encoded = json.dumps(identity, separators=(',', ':'), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], 'big')


def shift_seed(rep, role, width=0):
    record = DESIGN['replications'][str(rep)]
    batch, source_rep = record['batch'], record['source_rep']
    nonce = next((r['nonce'] for r in DESIGN['shift_seed_nonces']
                  if (r['batch'], r['rep'], r['role'], r['width']) ==
                  (batch, source_rep, role, width)), 0)
    return stream_seed('s102', 'development', batch, 'moderate', source_rep, role, width, nonce)


def fitting_seeds(study, case, rep, width):
    if study == 'shift':
        return shift_seed(rep, 'preparation', width), shift_seed(rep, 'candidate', width)
    r = DESIGN['replications'][str(rep)]
    namespace = DESIGN['namespaces'][r['batch']]
    return tuple(stream_seed(namespace, 'fitting', 5000, case, r['source_rep'], role, width)
                 for role in ('preparation:depth0', 'candidate:depth2'))


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
    seeds = DESIGN['replications'][str(rep)]['cases'][case]['training']
    result = {split:smooth(case, **seeds[split]) for split in SPLITS}
    if study == 'shift':
        for split, part in result.items():
            rng = np.random.default_rng(shift_seed(rep, 'outcome_'+split))
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
    seed = DESIGN['replications'][str(rep)]['cases'][case]['evaluation'][split]
    return smooth(case, 500 if split == 'validation' else 2500, seed)


def oob_counts(study, case, rep):
    from policy_core.oob import bootstrap_counts
    r = DESIGN['replications'][str(rep)]
    if study == 'shift':
        return bootstrap_counts(500, r['shift_oob_seed'])
    if rep < 20:
        # Original resampling draws are fixed design inputs, not observations.
        with np.load(Path(__file__).with_name('bootstrap_counts.npz')) as z:
            return z[f'{case}_{rep}'].astype(np.int64)
    return bootstrap_counts(500, r['cases'][case]['oob_seed'])
