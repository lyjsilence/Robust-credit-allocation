# KL-Calibrated Concept Drift: Table 2 Design

## Objective

Add a separate concept-drift experiment for the concept-drift columns of
Table 2. The experiment tests whether the robust policy's worst-environment
advantage changes as the deployment distribution moves farther from the
baseline distribution along a calibrated latent-Gaussian KL path.

The existing structural boundary-drift DGP and its results remain unchanged.
The new experiment is an additional stress test, not a replacement.

## Experimental design

- Use the existing nonlinear baseline DGP and historical training data.
- Train two policies per Monte Carlo replication:
  - the existing non-robust policy;
  - the existing robust policy with the Table 2 training radius
    `amb_radius = 0.2`.
- Train each policy once per replication. The training data do not depend on
  the deployment severity, so retraining at each severity would be redundant.
- Run 100 replications.
- Evaluate both policies at
  `lambda_2 = [0.5, 1, 1.5, 2, 2.5, 3]`.
- Use 100 drift environments per replication, matching the existing Table 2
  experiment.
- Use common covariates, latent errors, and drift directions across policies
  and severity levels within each replication.

This is a fixed-training-radius experiment. It does not use the matched-radius
training design proposed in the longer initial specification.

## New concept-drift DGP

Only the conditional latent-outcome distribution changes; the covariate
distribution remains fixed.

For each environment, construct a positive heterogeneous multiplier from a
small fixed basis of the existing covariates. Multiply the baseline treatment
contrast score by this multiplier, normalize the resulting two-action
direction, and shift the two latent Gaussian means in opposite directions.
The drift therefore attenuates the historical treatment margin without using
either learned policy to define the drift.

For severity `lambda_2`,

```text
kappa = 0.2 * lambda_2
eta_test = 0.5 * kappa^2
```

This gives:

| lambda_2 | eta_test |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

The normalization is computed on each replication's reference test sample, so
the empirical average latent-Gaussian KL equals `eta_test` up to numerical
precision. The same environment directions are reused at every severity,
forming nested drift paths.

## Components and data flow

1. A focused DGP module builds the heterogeneous drift direction and applies a
   requested severity to the existing baseline latent means.
2. A focused runner regenerates the validated baseline training/test draws,
   trains the two existing policy learners, obtains their actions once, and
   evaluates those actions over all severities and environments.
3. A small test module checks no-drift identity, covariate invariance,
   normalization, KL calibration, and increasing `eta_test`.

Drifted outcomes are created only in the evaluation stage and are never passed
to policy training.

## Evaluation and output

For each policy and environment, use the simulated potential outcomes directly:

```text
V(pi) = mean(Y_1 * 1{pi(X)=1} + Y_0 * 1{pi(X)=0})
```

For each replication and severity, retain the minimum value across the 100
environments. Aggregate these replication-level minima into a Table 2-style
CSV containing:

- `lambda_2` and `eta_test`;
- robust mean and standard error;
- non-robust mean and standard error;
- percentage improvement,
  `(robust_mean - nonrobust_mean) / nonrobust_mean * 100`.

Replication-level results are also saved for reproducibility. Models and
results go to a new, isolated output directory and never overwrite old
experiment artifacts.

## Verification and execution

The implementation intentionally avoids elaborate recovery and fallback code.
It will:

1. run syntax and focused DGP tests;
2. run a two-replication pilot in the `yijun_env` Conda environment;
3. inspect KL calibration, output structure, and finite policy values;
4. present the exact full-run command for confirmation;
5. run 100 replications with process-alive and timeout monitoring;
6. aggregate and report the resulting Table 2 values.

The expected increasing robust advantage is a hypothesis, not a programmed
success condition. Seeds and drift scales will not be changed after inspecting
the results to manufacture monotonicity.

## Scope exclusions

This implementation does not add matched training radii, radius grids, oracle
regret, figures, covariate-shift experiments, Table 1, Table 3, or automatic
parameter tuning.
