"""Generate observations for the complexity and perturbation experiments."""

from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

from .shift import SEVERITIES, nominal_outcomes

NORMALIZATION = {
    "low": (-0.0013596113120072362, 0.407926117841582),
    "moderate": (7.575993833541794e-05, 0.4083798384729396),
    "high": (-0.00016508049852892448, 0.40906725581128217),
}
SAMPLE_SIZES = {"train": 4500, "validation": 500, "test": 2500}
with np.load(Path(__file__).with_name("random_inputs.npz")) as archive:
    RANDOM_INPUTS = {name: archive[name] for name in archive.files}
CASES = ("low", "moderate", "high")
SPLITS = ("train", "validation", "test")


def shift_direction_seeds(rep):
    return RANDOM_INPUTS["shift_directions"][rep].tolist()


def shift_resampling_seed(rep, severity, environment):
    index = SEVERITIES[1:].index(severity)
    return int(RANDOM_INPUTS["shift_resampling"][rep, index, environment])


def fitting_seed(study, case, rep, width):
    if study == "shift":
        row = RANDOM_INPUTS["shift_fitting"][rep, width // 2 - 1]
    else:
        row = RANDOM_INPUTS["baseline_fitting"][rep, CASES.index(case), width // 2 - 1]
    return int(row)


def teacher(frequency):
    """Exact ReLU representation of the two-coordinate triangle score."""
    f = frequency
    net = nn.Sequential(nn.Linear(4, 6 * f), nn.ReLU(), nn.Linear(6 * f, 1)).double()
    with torch.no_grad():
        net[0].weight.zero_()
        net[0].bias.zero_()
        net[2].weight.zero_()
        net[2].bias.fill_(-1)
        for axis in range(2):
            for j in range(f):
                for k, (offset, coef) in enumerate(
                    ((0, 2 * f), (0.5, -4 * f), (1, 2 * f))
                ):
                    i = axis * 3 * f + 3 * j + k
                    net[0].weight[i, axis] = 0.5
                    net[0].bias[i] = 0.5 - (j + offset) / f
                    net[2].weight[0, i] = coef
    net.requires_grad_(False)
    return net.eval()


def smooth(case, n, seed):
    center, scale = NORMALIZATION[case]
    net = teacher(CASES.index(case) + 1)
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 4))
    with torch.no_grad():
        score = (net(torch.from_numpy(x)).numpy().ravel() - center) / scale
    delta = 1.5 * np.tanh(2 * score)
    outcomes = expit(np.column_stack((-delta, delta)) + rng.normal(0, 0.6, (n, 1)))
    nodes, weights = np.polynomial.hermite.hermgauss(40)
    offsets = np.sqrt(2) * 0.6 * nodes
    means = np.column_stack(
        [
            expit(sign * delta[:, None] + offsets) @ weights / np.sqrt(np.pi)
            for sign in (-1, 1)
        ]
    )
    propensity = np.full((n, 2), 0.5)
    actions = np.asarray([rng.choice(2, p=row) for row in propensity], dtype=np.int64)
    rows = np.arange(n)
    return dict(
        X=x.astype(np.float32),
        T=actions.reshape(-1, 1),
        Y=outcomes[rows, actions].reshape(-1, 1).astype(np.float32),
        prop_score=propensity[rows, actions].reshape(-1, 1).astype(np.float32),
        Y_all=outcomes.astype(np.float32),
        prop_score_all=propensity.astype(np.float32),
        mu=means.astype(np.float32),
    )


def training_data(study, case, rep):
    seeds = RANDOM_INPUTS["baseline_data"][rep, CASES.index(case)]
    result = {
        split: smooth(case, SAMPLE_SIZES[split], int(seeds[i]))
        for i, split in enumerate(SPLITS)
    }
    if study == "shift":
        for i, split in enumerate(SPLITS):
            part = result[split]
            rng = np.random.default_rng(int(RANDOM_INPUTS["shift_outcomes"][rep, i]))
            n = len(part["X"])
            g = nominal_outcomes(
                part["X"],
                part["mu"].argmax(1),
                rng.uniform(size=n),
                rng.normal(0, 0.6, size=n),
            )
            part["Y_all"] = g["Y_all"].astype(np.float32)
            part["Y"] = g["Y_all"][np.arange(n), part["T"].ravel(), None].astype(
                np.float32
            )
            part["mu"] = g["mu"].astype(np.float32)
            for key, source in (
                ("tail_p0", "p0"),
                ("tail_u", "U"),
                ("tail_z", "Z"),
                ("tail_risky_action", "risky_action"),
            ):
                part[key] = (
                    g[source]
                    .reshape(-1, 1)
                    .astype(np.int64 if key == "tail_risky_action" else np.float64)
                )
    return result


def held_out_data(case, rep, split):
    index = ("validation", "test").index(split)
    seed = int(RANDOM_INPUTS["baseline_evaluation"][rep, CASES.index(case), index])
    size = SAMPLE_SIZES[split]
    return smooth(case, size, seed)


def oob_counts(study, case, rep):
    from policy_core.evaluation import bootstrap_counts

    if study == "shift":
        return bootstrap_counts(500, int(RANDOM_INPUTS["shift_oob"][rep]))
    if rep < 20:
        return RANDOM_INPUTS["bootstrap_counts"][rep, CASES.index(case)].astype(
            np.int64
        )
    return bootstrap_counts(
        500, int(RANDOM_INPUTS["baseline_oob"][rep - 20, CASES.index(case)])
    )
