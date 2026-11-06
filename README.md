# Robust credit allocation: simulation experiments

Core code for PDRP policy learning with ordinary bootstrap out-of-bag (OOB)
calibration and for robust/non-robust comparisons under distribution shifts.
The two experiments below reproduce the specified Known-propensity PDRP row
and the covariate-shift/concept-drift table. They use different outcome models
and different evaluation functionals.

## Installation

Tested with Python 3.8.20, PyTorch 2.3.0+cu118, CUDA 11.8, Windows and an NVIDIA
RTX 3090. Install PyTorch separately from its CUDA wheel index:

```bash
python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Run commands from the repository root. `--device cpu` is supported, but exact
training reproducibility was checked on the environment above. Different
hardware or numerical-library versions can produce different fitted weights.

## Small training checks

Each command generates its observations from the fixed design and trains one
candidate from initialization. These are diagnostic fits, not complete width
selection or 100-replication experiments. Outputs must be new directories.

```bash
python -m simulation fit --study baseline --cases low --reps 0 --width 10 --output results/check_low
python -m simulation fit --study baseline --cases moderate --reps 0 --width 10 --output results/check_middle
python -m simulation fit --study baseline --cases high --reps 0 --width 8 --output results/check_high
python -m simulation fit --study shift --reps 0 --width 2 --method robust --output results/check_robust
python -m simulation fit --study shift --reps 0 --width 238 --method non-robust --output results/check_nominal
python -m simulation.verify --fit results/check_low
```

`model.pt` contains the fitted state and `complete.json` contains seeds and
optimization diagnostics. The verification command compares the model tensors
with a reference digest; it does not supply weights to the learner.

## Complete candidate pools and tables

```bash
# One replication, all three complexity cases, 128 widths per case:
python -m simulation run --study baseline --reps 0 --output results/baseline_rep0
python -m simulation.verify --run results/baseline_rep0

# One replication of the perturbation experiment, both methods:
python -m simulation run --study shift --reps 0 --output results/shift_rep0
python -m simulation.verify --run results/shift_rep0

# Full 100-replication experiments:
python -m simulation run --study baseline --all-reps --output results/baseline_100
python -m simulation run --study shift --all-reps --output results/shift_100
```

The default is replication 0. `--reps 0 1` requests a subset; `--cases low`
restricts the baseline case. A complete pool requires every even width from 2
through 256, so even a single-replication `run` is substantially longer than a
`fit`. Full experiments require 38,400 and 25,600 fits, respectively. Candidates
are trained sequentially with one CPU thread and the specified device. Existing
directories are refused, preventing accidental replacement of a completed or
partial run. Separate invocations may use separate output directories.

Each run writes a plan, per-candidate models, validation calibration arrays,
`selections_frozen.csv`, `replication_results.csv`, `summary.csv`, and a completion
record. Every requested selection is saved before test evaluation begins.
The perturbation run also saves all environment-level values, including the
supplemental severity 12. Only severities 0 through 10 belong to the displayed
table. The reference JSON is read only by verification, never by generation,
training, selection or evaluation.

## Baseline PDRP experiment

- Covariates: four independent Uniform(-1,1) features. The preferred-action
  score is a fixed two-coordinate ReLU triangle function with frequency 1, 2
  or 3 (Low, Middle or High), centered and scaled by fixed reference constants.
- Potential outcomes: logistic transforms of opposite signed score contrasts
  plus shared Gaussian noise with standard deviation 0.6. Logged actions have
  known probability 0.5 each.
- Fitting: 4,500 observations, two hidden ReLU layers, independently initialized
  candidates and one attempt per width. Robust training uses radius 0.2, Adam
  learning rate 0.005, weighted-classification pretraining and alternating
  policy/dual updates. The epoch ceilings, stopping rules and temperatures are
  in `simulation/design.json`; optimization checkpoints use fitting loss.
- OOB calibration uses a new, independent 500-row validation sample, 200
  size-500 bootstrap draws and a fixed 413-point coefficient grid. The training
  container's original validation data are not substituted for this sample.
- Width selection maximizes validation value minus `C * sqrt(width / 5000)`.
  The reported validation-scale coefficient is `c = C / sqrt(10)`. Ties within
  1e-10 choose the smaller width or coefficient.
- The displayed PDRP row selects within widths 2,4,...,128. Bootstrap validity
  is assessed across the entire 2,...,256 candidate pool. The same fixed
  resampling draws are used for every candidate in a replication.
- Evaluation uses 2,500 new observations and the self-normalized **observed
  outcome IPW KL-robust value**, with the evaluation dual reoptimized at radius
  0.2. It is not the full-potential-outcome estimator. Test regret is the best
  test value among the candidates up to width 128 minus the selected value;
  it is not regret against the population-optimal policy.

Across 100 replications, widths are medians and the uncertainty is the sample
standard deviation (`ddof=1`):

| Case | Median width | Value: mean ± SD | Regret: mean ± SD |
|---|---:|---:|---:|
| Low | 4 | 0.6309 ± 0.0069 | 0.0033 ± 0.0037 |
| Middle | 8 | 0.6214 ± 0.0092 | 0.0039 ± 0.0055 |
| High | 27 | 0.5721 ± 0.0182 | 0.0078 ± 0.0098 |

The median 27 is possible even though every fitted width is even. A small
subset has its own mean and spread and should not equal these 100-replication
statistics.

## Perturbation experiment

The Middle design's covariates and logged actions are reused with a continuous
lower-tail outcome mixture. One action usually has an outcome in (0.88,0.92)
but has an X-dependent probability of an outcome in (0,0.02); the other has an
outcome in (0.77,0.81). This differs from the baseline logistic-outcome model.

Robust candidates use radius 0.2; non-robust candidates optimize normalized IPW
means. They share data and initialization streams at each width. Each method
uses ordinary OOB with its matching objective to select from widths 2,...,256
using the 500 original held-out validation observations.

Covariate shift exponentially tilts the finite test covariate distribution
toward high-risk regions and resamples 2,500 observations. Concept drift keeps
covariates fixed and increases the lower-tail probability along random smooth
directions. The perturbation KL target is `0.5 * (0.2 * lambda)**2`, distinct
from the learner's radius 0.2. Each severity uses 100 paired environments.
Outcomes reuse their exogenous draws across paired methods/environments.

The per-replication score is the **minimum realized potential-outcome policy
mean over those 100 environments**, not an infimum over the entire KL ball.
The table reports mean ± standard error (`sample SD / sqrt(100)`) over the 100
replications. Improvement is `100 * (mean_robust - mean_nonrobust) /
mean_nonrobust`, calculated before rounding.

## Files and fixed design inputs

- `simulation/dgp.py`, `shift.py`: outcome generation and deployment shifts.
- `simulation/experiment.py`: fitting, selection, scoring and aggregation.
- `policy_core/`: numerical training, policy and KL/OOB routines.
- `simulation/design.json`: optimization settings, normalization and seeds.
- `simulation/bootstrap_counts.npz`: the fixed integer bootstrap multiplicities
  for the first 20 baseline replications. They contain no outcomes or model
  predictions. Later replications regenerate multiplicities from saved seeds.
- `reference_results.json`: table and replication-0 expectations plus five
  model-state digests, used only for verification.

Opaque stream identifiers and four recorded hash-collision nonces are retained
because changing them changes the random draws. No original filesystem layout,
private environment name, downloaded dataset or fitted checkpoint is required.
Empirical experiments, unrelated simulation variants, plots, manuscripts,
generated data/models and operational logs are not included.

## Verification scope

The release check regenerated 276 data arrays, retrained five candidate models,
reconstructed three baseline OOB pools and two perturbation OOB pools, and
recreated 1,400 paired perturbation environment rows. The trained model tensors
matched the retained references exactly; all checked selections and numerical
scores matched to 1e-12 or better. The stored 100-replication results were also
reaggregated and match both displayed tables at their reported precision.

This is a subset training check and a full saved-result aggregation check, not
a fresh 100-replication training run. Verification used the installed software
versions above in an isolated source directory, not a new operating-system or
package installation. These are development-selected simulation settings;
successful reproduction is not independent confirmation or proof of the
manuscript's theoretical claims.
