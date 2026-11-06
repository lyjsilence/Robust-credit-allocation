"""Generate observations for the complexity and perturbation experiments."""

from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from torch import nn


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
    from evaluation import bootstrap_counts

    if study == "shift":
        return bootstrap_counts(500, int(RANDOM_INPUTS["shift_oob"][rep]))
    if rep < 20:
        return RANDOM_INPUTS["bootstrap_counts"][rep, CASES.index(case)].astype(
            np.int64
        )
    return bootstrap_counts(
        500, int(RANDOM_INPUTS["baseline_oob"][rep - 20, CASES.index(case)])
    )


SEVERITIES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
KL_SCALE = 0.20
P_LOW = 0.005
P_HIGH = 0.780
NOISE_SD = 0.6


def target_kl(severity):
    severity = float(severity)
    return 0.5 * (KL_SCALE * severity) ** 2


def risk_score(x):
    x = np.asarray(x, dtype=np.float64)
    wave = np.sin(2 * np.pi * x[:, 0]) * np.sin(2 * np.pi * x[:, 1])
    return expit(30.0 * (wave - 0.70))


def baseline_probability(x):
    return P_LOW + (P_HIGH - P_LOW) * risk_score(x)


def nominal_outcomes(x, risky_action, uniforms, gaussian_noise):
    """Return continuous potential outcomes and fixed exogenous draws."""
    x = np.asarray(x, dtype=np.float64)
    risky_action = np.asarray(risky_action, dtype=np.int64).reshape(-1)
    uniforms = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    gaussian_noise = np.asarray(gaussian_noise, dtype=np.float64).reshape(-1)
    n = len(x)
    p0 = baseline_probability(x)
    z = expit(gaussian_noise)
    bad = 0.02 * z
    good = 0.88 + 0.04 * z
    stable = 0.77 + 0.04 * z
    risky = np.where(uniforms < p0, bad, good)
    y_all = np.empty((n, 2), dtype=np.float64)
    y_all[np.arange(n), risky_action] = risky
    y_all[np.arange(n), 1 - risky_action] = stable
    mu = np.empty_like(y_all)
    mu[np.arange(n), risky_action] = p0 * 0.01 + (1 - p0) * 0.90
    mu[np.arange(n), 1 - risky_action] = 0.79
    return {
        "Y_all": y_all,
        "mu": mu,
        "p0": p0,
        "U": uniforms,
        "Z": z,
        "risky_action": risky_action,
    }


def bernoulli_kl(p, p0):
    p = np.asarray(p, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    return p * np.log(p / p0) + (1 - p) * np.log((1 - p) / (1 - p0))


def drift_basis(x):
    x = np.asarray(x, dtype=np.float64)
    x1, x2 = x[:, 0], x[:, 1]
    nonconstant = np.column_stack(
        (x1, x2, x1 * x2, x1**2 - x2**2, np.sin(np.pi * x1), np.cos(np.pi * x2))
    )
    scales = nonconstant.std(axis=0)
    if np.any(scales <= 0):
        raise ValueError("Zero-scale drift feature")
    nonconstant = (nonconstant - nonconstant.mean(axis=0)) / scales
    return np.column_stack((np.ones(len(x)), nonconstant))


def susceptibility(x, direction_seeds):
    basis = drift_basis(x)
    rows = []
    for seed in direction_seeds:
        theta = np.random.default_rng(int(seed)).normal(size=basis.shape[1])
        theta /= np.linalg.norm(theta)
        rows.append(np.exp(0.35 * np.clip(basis @ theta, -2, 2)))
    return np.asarray(rows, dtype=np.float64)


def solve_concept_probability(p0, h, severity):
    """Calibrate each environment's average Bernoulli KL exactly."""
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1)
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    target = target_kl(severity)
    if target == 0:
        return p0.copy(), 0.0, 0.0
    baseline_logit = logit(p0)
    prob = lambda tau: expit(baseline_logit + tau * h)
    lo, hi = 0.0, 1.0
    while np.mean(bernoulli_kl(prob(hi), p0)) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Cannot bracket concept KL root")
    for _ in range(70):
        mid = (lo + hi) / 2
        if np.mean(bernoulli_kl(prob(mid), p0)) < target:
            lo = mid
        else:
            hi = mid
    tau = (lo + hi) / 2
    p = prob(tau)
    realized = float(np.mean(bernoulli_kl(p, p0)))
    if abs(realized - target) > 1e-9:
        raise AssertionError("Concept KL calibration failed")
    return p, float(tau), realized


def tilt_weights(x, severity):
    """Calibrate finite-sample covariate KL to the uniform test measure."""
    r = risk_score(x)
    target = target_kl(severity)
    n = len(r)
    if target == 0:
        return np.full(n, 1.0 / n), 0.0, 0.0

    def weights(beta):
        scores = beta * r
        unnormalized = np.exp(scores - scores.max())
        return unnormalized / unnormalized.sum()

    def kl(q):
        return float(np.sum(q * np.log(n * q)))

    lo, hi = 0.0, 1.0
    while kl(weights(hi)) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("Cannot bracket covariate KL root")
    for _ in range(70):
        mid = (lo + hi) / 2
        if kl(weights(mid)) < target:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2
    q = weights(beta)
    realized = kl(q)
    if abs(realized - target) > 1e-9:
        raise AssertionError("Covariate KL calibration failed")
    return q, float(beta), realized


def concept_outcomes(risky_action, uniforms, z, probabilities):
    """Potential outcomes for one fixed-X deployment environment."""
    risky_action = np.asarray(risky_action, dtype=np.int64).reshape(-1)
    uniforms = np.asarray(uniforms, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    n = len(risky_action)
    risky = np.where(uniforms < probabilities, 0.02 * z, 0.88 + 0.04 * z)
    stable = 0.77 + 0.04 * z
    outcomes = np.empty((n, 2), dtype=np.float64)
    outcomes[np.arange(n), risky_action] = risky
    outcomes[np.arange(n), 1 - risky_action] = stable
    return outcomes


def sample_path(data_dir, study, case, rep):
    return Path(data_dir) / study / case / f"rep_{rep:03d}" / "samples.npz"


def environment_path(data_dir, rep, mode, severity):
    folder = sample_path(data_dir, "shift", "moderate", rep).parent
    return folder / f"{mode}_level_{severity:g}.npz"


def load_samples(data_dir, study, case, rep):
    with np.load(sample_path(data_dir, study, case, rep)) as archive:
        return {
            split: {
                key.split("__", 1)[1]: archive[key]
                for key in archive.files
                if key.startswith(split + "__")
            }
            for split in SPLITS
        }


def generate_data(study, cases, reps, data_dir, levels=SEVERITIES, shift_mode="both"):
    """Save samples and requested deployment environments; reuse existing files."""
    modes = ("covariate", "concept") if shift_mode == "both" else (shift_mode,)
    for case in cases:
        for rep in reps:
            path = sample_path(data_dir, study, case, rep)
            if path.exists():
                data = load_samples(data_dir, study, case, rep)
            else:
                data = training_data(study, case, rep)
                if study == "baseline":
                    data["validation"] = held_out_data(case, rep, "validation")
                    data["test"] = held_out_data(case, rep, "test")
                path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {
                    f"{split}__{key}": array
                    for split, part in data.items()
                    for key, array in part.items()
                }
                np.savez_compressed(path, **arrays)
            print(f"Samples: {path}", flush=True)
            if study != "shift":
                continue
            test = data["test"]
            n = len(test["X"])
            if "concept" in modes:
                directions = susceptibility(test["X"], shift_direction_seeds(rep))
            for severity in levels:
                for mode in modes:
                    target = environment_path(data_dir, rep, mode, severity)
                    if target.exists():
                        print(f"Reusing {target}", flush=True)
                        continue
                    if mode == "covariate":
                        if severity == 0:
                            indices = np.tile(np.arange(n), (100, 1))
                        else:
                            weights, _, _ = tilt_weights(test["X"], severity)
                            indices = np.stack(
                                [
                                    np.random.default_rng(
                                        shift_resampling_seed(rep, severity, m)
                                    ).choice(n, n, replace=True, p=weights)
                                    for m in range(100)
                                ]
                            )
                        np.savez_compressed(target, indices=indices)
                    else:
                        if severity == 0:
                            outcomes = np.repeat(test["Y_all"][None, :, :], 100, axis=0)
                        else:
                            outcomes = []
                            for direction in directions:
                                probability, _, _ = solve_concept_probability(
                                    test["tail_p0"].ravel(), direction, severity
                                )
                                outcomes.append(
                                    concept_outcomes(
                                        test["tail_risky_action"].ravel(),
                                        test["tail_u"].ravel(),
                                        test["tail_z"].ravel(),
                                        probability,
                                    ).astype(np.float32)
                                )
                            outcomes = np.stack(outcomes)
                        np.savez_compressed(target, outcomes=outcomes)
                    print(f"Environments: {target}", flush=True)
