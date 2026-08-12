# Middle-Margin Covariate-Shift DGP Design

## Goal

Replace the random covariate-shift score with a predefined function of the DGP
covariates that concentrates deployment mass where policy decisions remain
difficult but choosing the wrong action already has a material outcome cost.
The design must remain a pure covariate shift and preserve the six requested KL
levels.

## Predefined shift score

Let \(s(X_1,X_2)\) be the score defining the nominal treatment boundary and let
\(\gamma\) be the sharpness parameter already used in the smooth-outcome DGP.
Define

\[
h(X)=|s(X_1,X_2)|\exp\{-\gamma|s(X_1,X_2)|\}.
\]

This score is determined before policy learning and uses neither outcomes,
learned policies, nor previous simulation results. It equals zero at the exact
decision boundary, reaches its maximum at \(|s(X)|=1/\gamma\), and decreases in
regions far from the boundary. It therefore emphasizes an intermediate-margin
region rather than observations that are either outcome-indifferent or easy to
classify.

For the nonlinear Table 2 DGP,

\[
s(X_1,X_2)=X_1-b_R(X_2),
\]

where \(b_R\) is the existing S-shaped boundary. The unified generator applies
the same principle to the linear score when that branch is used.

## Shift distribution

For each nominal test sample, define

\[
q_i(\lambda_1)
=
\frac{\exp\{\beta(\lambda_1)h(X_i)\}}
{\sum_j\exp\{\beta(\lambda_1)h(X_j)\}}.
\]

For each replication and severity, a scalar \(\beta(\lambda_1)\geq0\) is found
by bisection so that

\[
D_{\mathrm{KL}}\{q(\lambda_1)\|p_0\}=\eta(\lambda_1),
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

Each of the 100 deployment environments independently resamples 2,500 rows
from the same calibrated distribution. The existing saved-data contract
(`base`, `envs`, `q_list`, and `kl_list`) remains unchanged.

## Scope

- Modify only the covariate-shift branch of `simulation/generate_dgp.py`.
- Keep the nominal smooth-outcome equation, behavior policy, sample sizes,
  random seeds, filenames, and output structure unchanged.
- Keep the concept-drift DGP exactly unchanged.
- Keep policy learning, the training ambiguity radius of 0.2, and
  worst-environment evaluation unchanged.
- Remove the random-normal shift score from the revised covariate branch.
- Do not introduce tunable center or bandwidth parameters: the score peak is
  fixed by the existing outcome sharpness \(\gamma\).

## Validation

1. Verify exact equality of nominal and concept generation before and after the
   code change.
2. Verify all six realized KL values against their targets.
3. Verify that the weighted mean of \(h(X)\) increases with \(\lambda_1\).
4. Verify that resampling keeps all row-level arrays aligned.
5. Run the complete DGP test suite.
6. Generate an isolated five-replication pilot and train each policy once per
   replication on the common nominal training sample.
7. Evaluate each fixed policy on all six shifted test sets and report the full
   pilot sequence.

The score definition will not be changed in response to pilot results. If the
pilot does not show the intended mechanism, its temporary code, data, models,
and results will be removed before another design is considered. Formal Table 2
data and results will not be overwritten by a pilot.

## Interpretation and limitation

The revised design changes only \(P_X\); it leaves \(P(Y\mid X,A)\) and the
behavior policy conditional on \(X\) unchanged. It is therefore a covariate
shift. The middle-margin construction provides an ex ante reason that policy
mistakes may become more consequential without making classification trivial.
It does not guarantee that the robust improvement will be monotone in a finite
simulation, and any such pattern must be established by the prespecified
experiment rather than assumed from the design.
