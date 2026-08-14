# Structured covariate-shift experiment

This folder is an isolated search experiment. It does not read from or write to
the formal `simulation/data`, `simulation/models`, or `simulation/results`
directories. Policy learning and evaluation are imported unchanged from the
formal simulation modules.

## Selected DGP

Covariates are sampled uniformly from the five-dimensional unit ball. The
nonlinear decision boundary based on `(X1, X2)` determines which of the two
actions is risky. The other action has the stable outcome `0.50`. The risky
action has outcome `0.95` unless a catastrophe produces outcome zero.

The catastrophe probability is

```text
v(X) = sigmoid(20 * [sin(2 pi X3) sin(2 pi X4) - 0.10])
p_cat(X) = 0.02 + (0.90 - 0.02) v(X).
```

Thus, `Y | X, T` is fixed across environments. Covariate shift is introduced
only by exponentially tilting the test distribution toward large `v(X)`:

```text
q_lambda(X_i) proportional to exp(beta_lambda * v(X_i)).
```

For each replication and lambda, `beta_lambda` is calibrated so that the
realized KL divergence from the unshifted empirical test distribution equals
`0.5 * (0.2 * lambda)^2`. The six target KL values are therefore `0.005`,
`0.02`, `0.045`, `0.08`, `0.125`, and `0.18`.

## Reproduce

Run from the repository root in the `yijun_env` Conda environment:

```powershell
conda run -n yijun_env python simulation\shift_dgp_search\dgp.py `
  --candidate islands_k2_t10 --num_reps 100

conda run -n yijun_env python simulation\shift_dgp_search\run_experiment.py `
  --candidate islands_k2_t10 --num_reps 100

conda run -n yijun_env python simulation\shift_dgp_search\summarize.py `
  --candidate islands_k2_t10 --stage full100 --num_reps 100
```

The ambiguity radius is `0.2`, as in the formal Table 2 experiment. Generated
data, models, and result PKLs remain local because they are ignored by Git.

## 100-replication result

| lambda | KL | Robust | Non-robust | Improvement of means |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.005 | 0.498714 | 0.522984 | -4.64% |
| 1.0 | 0.020 | 0.496702 | 0.489179 | 1.54% |
| 1.5 | 0.045 | 0.494808 | 0.454894 | 8.77% |
| 2.0 | 0.080 | 0.492753 | 0.422191 | 16.71% |
| 2.5 | 0.125 | 0.490942 | 0.388734 | 26.29% |
| 3.0 | 0.180 | 0.489017 | 0.356607 | 37.13% |

The mean paired improvements are `-4.62%`, `1.57%`, `8.83%`, `16.81%`,
`26.45%`, and `37.38%`. Their respective 95% confidence intervals are
`[-5.01, -4.22]`, `[1.15, 2.00]`, `[8.32, 9.34]`, `[16.14, 17.48]`,
`[25.57, 27.32]`, and `[36.22, 38.54]` percentage points.

## Research note

This DGP was selected after exploratory comparison with other candidate
frequencies and thresholds. The complete attempt-level record is retained in
`attempts.csv`. Consequently, the result should be described as an exploratory
DGP design result, not as a pre-specified confirmatory test.
