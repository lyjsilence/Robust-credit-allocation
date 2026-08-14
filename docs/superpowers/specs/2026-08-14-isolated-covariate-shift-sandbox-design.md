# Isolated Covariate-Shift DGP Sandbox

## Goal and isolation boundary

Create an exploratory covariate-shift experiment without changing the current
simulation code, data, models, or results. All new executable code and generated
artifacts will live below:

```text
simulation/shift_dgp_sandbox/
```

The sandbox may read the existing policy-learning and policy-evaluation modules,
but it must not write to `simulation/data`, `simulation/models`, or
`simulation/results`. A failed experiment can therefore be removed by deleting
only the sandbox directory.

## Experiment grid and success criterion

The experiment will use 100 paired replications at:

| lambda_1 | Target KL |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

The primary success criterion is based on paired percentage improvement,

```text
100 * (Q_robust - Q_nonrobust) / Q_nonrobust.
```

The six mean improvements must be nondecreasing. The improvement at
`lambda_1=3` must be at least 10%, and the increase from `lambda_1=0.5` to
`lambda_1=3` must be at least 8 percentage points.

## Primary DGP: fixed X-dependent lower-tail risk

Covariates remain uniformly distributed on the five-dimensional unit ball. The
nonlinear score in `(X1, X2)` determines which of the two actions is the risky
action. A separate vulnerability surface is defined by

```text
z(X) = 1.4 X3 - 0.9 X4 + 0.8 X3 X5 - 0.35
v(X) = sigmoid(8 z(X)).
```

The stable action has outcome `0.79`. The risky action has outcome `0.95` with
probability `1 - p(X)` and outcome `0` otherwise, where

```text
p(X) = 0.04 + 0.20 v(X).
```

The logging policy assigns the risky action with probability `0.75` and the
stable action with probability `0.25`. These conditional outcome and logging
mechanisms are identical in training and every test environment.

For each severity, the test covariate weights are

```text
q_lambda(X) proportional to exp(beta_lambda * v(X)),
```

where `beta_lambda` is solved so the empirical KL divergence from the uniform
test distribution equals the target value. Only `P(X)` changes across test
environments, so this remains a covariate-shift DGP.

## Pre-specified fallback: weak overlap in the vulnerable subgroup

The fallback retains the same covariates, vulnerability surface, outcomes, KL
targets, and test shift. It changes only the training logging probability of the
risky action to

```text
0.75 + 0.20 v(X),
```

with the known propensity recorded in the data. This produces weaker overlap
for the stable action in the vulnerable subgroup. The fallback will be tried
only if the primary pilot fails; no further DGP variants or parameter tuning are
in scope.

## Files and data flow

The sandbox will contain two readable scripts:

```text
shift_dgp_sandbox/
  generate_dgp.py
  run_experiment.py
  data/
  models/
  results/
```

`generate_dgp.py` will generate one base file per replication. Each file will
store the training data, the unshifted test base, and resampling indices and KL
metadata for all six severities. Storing indices rather than six copies of every
environment keeps the sandbox compact.

`run_experiment.py` will import the existing policy-learning and evaluation
functions. For each replication and method, it will train one policy because the
training distribution is common to every severity, then evaluate that same
policy on all six shifted test distributions. Models and result PKLs will be
written only below the sandbox. Result directory and filename conventions will
match the main simulation so the existing `simulation/concat_results.py` can
summarize the sandbox by passing `--results_dir` and `--output`.

## Staged execution

1. Generate and run a 10-replication primary pilot at `lambda_1=0.5, 1.5, 3`.
2. Continue to the full six-level, 100-replication primary experiment only if
   the pilot is monotone, reaches at least 8% at `lambda_1=3`, and increases by
   at least 6 percentage points from `lambda_1=0.5`.
3. If the primary pilot fails, remove its generated artifacts and run the same
   10-replication pilot for the pre-specified fallback.
4. Run the fallback at 100 replications only if it passes the same pilot gate.
5. Apply the full success criterion to the 100-replication summary. If neither
   variant succeeds, delete `simulation/shift_dgp_sandbox` and leave the current
   formal experiment unchanged.

## Verification

Before policy training, verify for representative replications that every
realized KL equals its target within numerical tolerance and that training data
are identical across severity evaluations. During experiment execution, monitor
process liveness and output counts. After completion, calculate paired mean
improvements and 95% confidence intervals and confirm that no formal simulation
file changed.

This sandbox is exploratory. If a candidate is retained for the paper, its
mechanism must be justified independently of the observed result and the
selection process should be documented as part of the simulation-development
record.
