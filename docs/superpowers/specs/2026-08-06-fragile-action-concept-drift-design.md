# Fragile-Action Concept Drift: Table 2 Design

## Objective

Create a separate Table 2 concept-drift experiment in which the historical
data contain a policy-independent mean-versus-downside-risk trade-off. The
experiment tests whether the existing KL-robust learner selects less fragile
actions and consequently loses less value as deployment drift becomes more
severe.

The original structural drift and the first latent-Gaussian margin-attenuation
experiment remain unchanged.

## Historical DGP

Keep the original five-dimensional covariate distribution and nonlinear score.
For each observation, the sign of the nonlinear score defines the fragile,
high-return action `a_H(X)`; the other action is the stable action `a_L(X)`.

The baseline latent potential outcomes are:

```text
Z_fragile = mu_fragile(X) + sigma_fragile * epsilon_fragile
Z_stable  = mu_stable      + sigma_stable  * epsilon_stable
Y_a       = sigmoid(Z_a)
```

Use the fixed parameters:

```text
mu_stable       = 0.60
mu_fragile(X)   = 0.80 + 0.20 * abs(tanh(6 * score(X)))
sigma_stable    = 0.20
sigma_fragile   = 0.80
```

Thus, the fragile action has a higher baseline expected outcome but a
substantially worse lower tail. The logging policy chooses the fragile action
with probability `0.75`, matching the original overlap level.

These parameters are fixed before policy training. They will not be changed
after inspecting robust-versus-non-robust performance.

## Deployment concept drift

The covariate distribution, action assignment mechanism, and latent variances
remain unchanged. Only the fragile action's conditional latent mean
deteriorates:

```text
mu_fragile_drift(X)
    = mu_fragile(X) - kappa * sigma_fragile * v_m(X)

mu_stable_drift(X)
    = mu_stable
```

For environment `m`, construct a positive heterogeneous function from the same
small covariate basis used in the first new experiment:

```text
h_m(X) = exp(0.35 * clip(theta_m' phi(X), -2, 2))
v_m(X) = h_m(X) / sqrt(mean(h_m(X)^2))
```

The same 100 environment directions are reused at every severity. Covariates,
latent errors, and directions are common across both policies and all
severities within a replication.

Use:

```text
kappa = 0.20 * lambda_2
eta_test = 0.5 * kappa^2
```

Because only one action is shifted and `mean(v_m^2) = 1`,

```text
0.5 * mean((Delta_mu_fragile / sigma_fragile)^2) = eta_test.
```

The requested mapping remains:

| lambda_2 | eta_test |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

## Policy training and evaluation

For each of 100 replications:

1. Generate one new historical training dataset with 10,000 observations.
2. Train the existing robust policy with `amb_radius = 0.2`.
3. Train the existing non-robust policy using the same historical DGP.
4. Generate one 2,500-observation reference test sample.
5. Evaluate both fixed policies over 100 deployment environments at all six
   severity levels using true simulated potential outcomes.
6. Retain each policy's minimum environment value, matching the original
   Table 2 evaluation.

The two methods retain the original training hyperparameters and random-seed
protocol. Drifted outcomes are never passed to policy training.

## Pre-registered manipulation checks

Before interpreting policy performance, verify:

1. The fragile action's baseline mean outcome exceeds the stable action's mean
   outcome by at least `0.02`.
2. The fragile action's baseline 10th-percentile outcome is at least `0.05`
   below the stable action's 10th percentile.
3. In the two-replication pilot, the robust policy's fragile-action exposure
   is at least five percentage points lower than the non-robust policy's
   exposure on average.

Checks 1 and 2 validate the DGP independently of learned policy results.
Check 3 verifies that the intended risk trade-off is operational for the
existing learner. If check 3 fails, stop and report that this DGP does not test
the intended mechanism; do not tune parameters automatically.

## Outputs

Write to a new isolated directory:

```text
results/fragile_action_concept_drift/
```

Save:

- replication-level minimum values and fragile-action exposures;
- a Table 2-style CSV with robust mean and standard error, non-robust mean and
  standard error, and percentage improvement;
- a short manipulation-check CSV.

No figures, radius grids, other tables, or unrelated diagnostics are included.

## Verification and execution

Add focused tests for:

- no-drift identity;
- unchanged covariates and variances;
- baseline mean/lower-tail trade-off;
- only the fragile action being shifted;
- empirical KL calibration;
- increasing severity mapping.

Then run a two-replication pilot in `yijun_env`. If the manipulation checks
pass, present the exact command for confirmation before running 100
replications. Results are reported without changing seeds or DGP parameters
after performance inspection.
