# Unified tail-risk sandbox

This isolated experiment places covariate shift and concept drift in the same
probabilistic outcome family. It does not modify the formal simulation data,
models, results, policy learner, or evaluator.

## Shared outcome model

The existing nonlinear boundary in `(X1, X2)` identifies a risky action. The
other action is stable. Both modes use:

```text
stable outcome       = 0.79
good risky outcome   = 0.90
bad risky outcome    = 0.00
P(logged risky action | X) = 0.75
```

The risky outcome equals `0.90` unless a Bernoulli catastrophe produces zero.

## Covariate shift

The fixed conditional catastrophe probability is

```text
v(X) = sigmoid(30 * [sin(2 pi X3) sin(2 pi X4) - 0.70])
p0(X) = 0.02 + 0.76 v(X).
```

Its nominal population mean is approximately `0.10`. Deployment covariates are
exponentially tilted toward large `v(X)`, while the conditional outcome model is
held fixed. Thus only `P(X)` changes.

## Concept drift

The baseline catastrophe probability is `0.10`. Deployment environments alter
this probability on the logit scale using the same X-dependent susceptibility
as the formal concept-drift DGP, while `P(X)` remains fixed. Replication 0 at
lambda 0.5 was checked against the formal data: training data, test base, and
all 100 deployment environments matched exactly for `Y`, `Y_all`, and `mu`.

## KL calibration

Both modes use the target `0.5 * (0.2 * lambda)^2`, giving KL values `0.005`,
`0.02`, `0.045`, `0.08`, `0.125`, and `0.18`. Across 100 replications, the
largest absolute calibration errors were `3.40e-16` for covariate shift and
`1.39e-16` for concept drift.

## 100-replication results

### Covariate shift

| lambda | KL | Robust | Non-robust | Improvement of means |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.005 | 0.786087 | 0.779356 | 0.86% |
| 1.0 | 0.020 | 0.784499 | 0.762700 | 2.86% |
| 1.5 | 0.045 | 0.782419 | 0.745307 | 4.98% |
| 2.0 | 0.080 | 0.780520 | 0.727005 | 7.36% |
| 2.5 | 0.125 | 0.778392 | 0.708150 | 9.92% |
| 3.0 | 0.180 | 0.776102 | 0.688715 | 12.69% |

The paired mean improvements and 95% confidence intervals are `0.87%
[0.67, 1.07]`, `2.87% [2.62, 3.12]`, `5.01% [4.66, 5.35]`, `7.40%
[6.97, 7.84]`, `9.99% [9.43, 10.54]`, and `12.79% [12.10, 13.48]`.

### Concept drift

| lambda | KL | Robust | Non-robust | Improvement of means |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.005 | 0.788342 | 0.781764 | 0.84% |
| 1.0 | 0.020 | 0.785232 | 0.757514 | 3.66% |
| 1.5 | 0.045 | 0.781887 | 0.732538 | 6.74% |
| 2.0 | 0.080 | 0.778392 | 0.706229 | 10.22% |
| 2.5 | 0.125 | 0.774767 | 0.678934 | 14.12% |
| 3.0 | 0.180 | 0.770965 | 0.650512 | 18.52% |

The paired mean improvements and 95% confidence intervals are `0.85%
[0.71, 0.99]`, `3.67% [3.49, 3.85]`, `6.75% [6.50, 7.00]`, `10.24%
[9.88, 10.60]`, `14.16% [13.67, 14.65]`, and `18.59% [17.95, 19.22]`.

## Reproduce

Run from the repository root:

```powershell
conda run -n yijun_env python simulation\shift_dgp_search\unified_tail_risk\dgp.py `
  --mode covariate --num_reps 100
conda run -n yijun_env python simulation\shift_dgp_search\unified_tail_risk\dgp.py `
  --mode concept --num_reps 100

conda run -n yijun_env python simulation\shift_dgp_search\unified_tail_risk\run_experiment.py `
  --dgp_mode covariate --num_reps 100
conda run -n yijun_env python simulation\shift_dgp_search\unified_tail_risk\run_experiment.py `
  --dgp_mode concept --num_reps 100
```

Generated data, models, and result PKLs are ignored by Git. `summary.csv`
contains both the pilot and full paired summaries. The covariate specification
is exploratory because its risk threshold was selected after analytical
screening.
