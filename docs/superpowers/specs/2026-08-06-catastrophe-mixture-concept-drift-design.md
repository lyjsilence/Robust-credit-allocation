# Catastrophe-Mixture Concept Drift: Table 2 Design

## Objective

Add one self-contained Table 2 concept-drift experiment in which the nominal
and robust population objectives prefer different actions for a transparent,
pre-specified reason. The historical DGP contains a high-mean action with a
small catastrophe probability and a lower-mean stable action. Deployment drift
increases only the catastrophe probability while keeping the covariate
distribution fixed.

The two unsuccessful experimental implementations were removed from the
current working tree. The original paper DGP, data, runner, and results remain
unchanged.

## Historical DGP

Keep the original five-dimensional uniform-ball covariates and nonlinear score.
The sign of the score defines which action is the fragile high-return action
`a_H(X)`; the other action is stable.

Use a latent catastrophe indicator:

```text
C ~ Bernoulli(p_0)
p_0 = 0.05

Y_fragile = 0.05 if C = 1 else 0.78
Y_stable  = 0.66
```

Equivalently, these values can be represented by fixed latent logits followed
by the existing sigmoid transform. The catastrophe indicator is independent
across observations and actions are logged with:

```text
P(T = a_H(X) | X) = 0.75.
```

The fragile action's population mean is:

```text
0.05 * 0.05 + 0.95 * 0.78 = 0.7435 > 0.66.
```

Thus, the population nominal policy prefers the fragile action.

At KL radius `0.2`, shifting the catastrophe probability from `0.05` toward
approximately `0.24` lowers the fragile action's worst-case value to about
`0.605`, below the stable value `0.66`. Thus, the population KL-robust
criterion prefers the stable action.

All values and probabilities are fixed before policy training and will not be
changed after inspecting policy performance.

## Deployment concept drift

Only the fragile action's conditional catastrophe probability changes.
Covariates, the stable action, and the logging mechanism remain unchanged.

For environment `m`, construct a positive susceptibility function:

```text
h_m(X) = exp(0.35 * clip(theta_m' phi(X), -2, 2)).
```

The basis `phi(X)` is the small fixed basis:

```text
1, X_1, X_2, X_1 * X_2, X_1^2 - X_2^2,
sin(pi * X_1), cos(pi * X_2).
```

For severity `lambda_2`, define:

```text
eta_test = 0.5 * (0.20 * lambda_2)^2

p_m,lambda(X)
    = logistic(logit(p_0) + tau_m,lambda * h_m(X)).
```

For every environment and severity, solve the single nonnegative scalar
`tau_m,lambda` so that:

```text
mean_X[
    KL(Bernoulli(p_m,lambda(X)) || Bernoulli(p_0))
] = eta_test.
```

Because the catastrophe outcome and non-catastrophe outcome are distinct, the
conditional outcome KL is exactly the Bernoulli KL. No latent-versus-observed
KL approximation is required.

The requested mapping remains:

| lambda_2 | eta_test |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

## Common random numbers

For each replication:

1. Draw training and test covariates once.
2. Draw one uniform catastrophe variable per test observation.
3. Reuse the same uniforms across all environments and severities.
4. Reuse the same 100 environment directions at every severity.

Since `p_m,lambda(X)` increases with severity, catastrophe events are nested
along each environment path.

## Training and evaluation

For each replication:

1. Generate 10,000 historical observations from the baseline mixture.
2. Train the existing robust policy with `amb_radius = 0.2`.
3. Train the existing non-robust policy with the same settings as Table 2.
4. Generate a 2,500-observation test sample.
5. Evaluate the two fixed policies over 100 environments at all six
   severities using simulated potential outcomes.
6. Retain the minimum environment value for each policy, matching Table 2.

Run two replications first. Run 100 replications only after the pre-registered
pilot checks pass and the exact command is confirmed.

## Pre-registered checks

Before interpreting the performance trend, verify:

1. Baseline fragile mean exceeds stable mean by at least `0.05`.
2. At radius `0.2`, the population KL worst-case stable value exceeds the
   fragile value by at least `0.03`.
3. Every environment-severity empirical KL differs from `eta_test` by less
   than `1e-8`.
4. In the two-replication pilot, non-robust fragile-action exposure exceeds
   robust exposure by at least five percentage points.

If check 4 fails, stop and report the failure. Do not change the mixture
values, catastrophe probability, seed, or shift scale automatically.

The robust advantage is expected to increase with severity, but monotonicity
is not a programmed acceptance condition.

## Files and outputs

Create only:

```text
catastrophe_mixture_concept_drift.py
run_catastrophe_mixture_concept_drift.py
tests/test_catastrophe_mixture_concept_drift.py
```

Write all models and results to a new ignored directory:

```text
results/catastrophe_mixture_concept_drift/
```

Save:

- replication-level policy values and fragile-action exposures;
- Table 2-style aggregate results;
- environment-level realized KL checks;
- a short manipulation-check summary.

No figures, radius grids, policy-targeted drift, or unrelated refactoring are
included.
