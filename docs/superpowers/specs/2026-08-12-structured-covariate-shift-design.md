# Structured Covariate-Shift DGP Design

## Goal

Replace the random covariate-shift score with a predefined score that depends
only on the nonlinear DGP geometry. The revised shift should remain a pure
covariate shift, preserve the six requested KL levels, and provide an
interpretable mechanism through which larger shifts can reveal differences
between robust and non-robust policies.

## Shift score

For the nonlinear score

\[
s(x_1,x_2)=x_1-b_R(x_2),
\]

use

\[
h(x)=|s(x_1,x_2)|.
\]

The score is fixed before policy learning and uses neither outcomes nor learned
policies. Larger values identify observations farther from the nonlinear
decision boundary, where choosing the wrong action has a larger outcome cost.
Using the absolute value treats the two action regions symmetrically and avoids
selecting a shift direction based on action labels.

## Shift distribution

For each nominal test sample, define

\[
q_i(\lambda_1)
=
\frac{\exp\{\beta(\lambda_1)h(X_i)\}}
{\sum_j\exp\{\beta(\lambda_1)h(X_j)\}}.
\]

For each replication and severity, solve for the nonnegative scalar
\(\beta(\lambda_1)\) so that

\[
D_{\mathrm{KL}}(q(\lambda_1)\|p_0)=\eta(\lambda_1),
\]

where \(p_0\) is uniform on the nominal test sample and

| \(\lambda_1\) | Target KL \(\eta(\lambda_1)\) |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

A short bisection routine is sufficient because the KL divergence is monotone
in \(\beta\). Each of the 100 deployment environments independently resamples
2,500 observations from the same structured distribution \(q(\lambda_1)\).
The existing saved-data schema (`base`, `envs`, `q_list`, and `kl_list`) is
retained; `q_list` contains the common calibrated weights for compatibility.

## Scope

- Modify only the covariate-shift branch of `simulation/generate_dgp.py`.
- Keep the nominal nonlinear DGP, smooth outcome equation, behavior policy,
  random seeds, sample sizes, and file naming unchanged.
- Keep concept drift unchanged.
- Keep policy learning, the robust training radius of 0.2, and
  worst-environment evaluation unchanged.
- Do not use learned policies, observed outcomes, or previous Table 2 results
  when defining or calibrating the shift.
- Remove the random normal shift score from the covariate branch because it is
  no longer part of the revised DGP.

## Validation

1. Verify that nominal generated data remain exactly unchanged.
2. Verify all six realized KL values against their targets.
3. Verify that mean \(|s(X)|\) increases monotonically with \(\lambda_1\).
4. Verify that every resampled row keeps `X`, `T`, `Y`, `Y_all`, `mu`, and
   propensity entries aligned.
5. Verify that concept-drift generation remains unchanged.
6. Run a small pilot using the existing experiment pipeline. If the expected
   mechanism is present, regenerate all six covariate datasets and rerun the
   100-replication Table 2 covariate panel. The DGP definition will not be
   selected or changed based on pilot outcomes.

## Interpretation

The revised design changes only \(P_X\). Conditional outcomes and treatment
assignment mechanisms remain fixed, so it is still a covariate-shift experiment.
As severity increases, deployment mass moves toward observations with larger
treatment-effect contrast. This makes policy errors more consequential and
provides an ex ante, policy-independent reason for differences in worst-case
policy value to become more visible.
