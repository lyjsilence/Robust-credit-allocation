# Catastrophe-Mixture Concept Drift: Final Table 2 Design

## Objective

Run the existing robust and non-robust policy learners under a transparent
concept-drift DGP in which deployment drift progressively increases lower-tail
risk. Keep the Table 2 training radius, sample sizes, number of environments,
severity grid, and policy-learning code unchanged.

The final DGP was frozen only after passing the discovery and independent
holdout gates in
`2026-08-06-dgp-exploration-protocol-design.md`.

## Historical DGP

Covariates follow the original five-dimensional uniform-ball design. The sign
of the original nonlinear score determines the risky high-return action
`a_R(X)`; the other action is stable.

```text
C ~ Bernoulli(0.10)

Y_risky  = 0.00 if C = 1 else 0.90
Y_stable = 0.79
```

The risky action has nominal population mean

```text
0.10 * 0.00 + 0.90 * 0.90 = 0.81,
```

which exceeds the stable outcome by `0.02`. The logging policy selects the
risky action with probability `0.75`.

At KL radius `0.2`, the risky action's population worst-case value is about
`0.60075`, so the stable action has a worst-case advantage of about `0.18925`.

## Deployment concept drift

Only the risky action's catastrophe probability changes. Covariates, the
stable outcome, and the logging mechanism remain unchanged.

For environment `m`, define

```text
h_m(X) = exp(0.35 * clip(theta_m' phi(X), -2, 2)),
```

using the fixed basis

```text
1, X_1, X_2, X_1 * X_2, X_1^2 - X_2^2,
sin(pi * X_1), cos(pi * X_2).
```

For severity `lambda_2`,

```text
eta_test = 0.5 * (0.20 * lambda_2)^2,

p_m,lambda(X)
    = logistic(logit(0.10) + tau_m,lambda * h_m(X)).
```

For every environment and severity, solve the nonnegative scalar
`tau_m,lambda` so the mean conditional Bernoulli KL equals `eta_test`.

| lambda_2 | eta_test |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

One catastrophe uniform is reused across all environments and severities for
each test observation. Catastrophes are therefore nested as severity rises.

## Training and evaluation

For each of 100 replications:

1. Generate 10,000 historical training observations.
2. Train the existing robust policy at radius `0.2`.
3. Train the existing non-robust policy with the same Table 2 settings.
4. Generate 2,500 test observations.
5. Evaluate both fixed policies in 100 environments at all six severities.
6. Retain the minimum environment value for each policy.

The robust estimator, optimizer, neural-network classes, and training
hyperparameters are not modified.

## Files and outputs

Journal code:

```text
simulation/catastrophe_mixture_concept_drift.py
simulation/run_catastrophe_mixture_concept_drift.py
simulation/tests/test_catastrophe_mixture_concept_drift.py
```

Final ignored output directory:

```text
simulation/results/catastrophe_mixture_concept_drift_100rep/
```

The output contains replication-level values, the Table 2 aggregate, KL
checks, manipulation checks, logs, and fitted models.
