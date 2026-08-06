# Covariate-Shift KL Calibration Design

## Goal

Calibrate the covariate-shift parameter so that

| \(\lambda_1\) | Target mean realized KL |
|---:|---:|
| 0.5 | 0.005 |
| 1.0 | 0.020 |
| 1.5 | 0.045 |
| 2.0 | 0.080 |
| 2.5 | 0.125 |
| 3.0 | 0.180 |

while preserving the existing covariate outcome model, random seeds, saved-data
schema, policy learner, and training ambiguity radius of 0.2.

## DGP Change

Use one shared constant,

```python
SHIFT_SCALE = 0.20
```

and define the covariate-shift sampling weights by

\[
q_{im}(\lambda_1)
=
\frac{\exp\{SHIFT\_SCALE\cdot\lambda_1 z_{im}\}}
{\sum_j\exp\{SHIFT\_SCALE\cdot\lambda_1 z_{jm}\}},
\qquad z_{im}\sim N(0,1).
\]

For standard-normal exponential tilting,

\[
D_{\mathrm{KL}}(q_m\|p_0)
\approx
\frac{1}{2}(SHIFT\_SCALE\cdot\lambda_1)^2,
\]

which gives the six requested targets. The concept-drift calibration will reuse
the same constant; its DGP is otherwise unchanged.

## Scope

- Change only the strength of the covariate resampling weights.
- Keep the smooth covariate-shift outcome equation unchanged.
- Keep the six displayed \(\lambda_1\) labels unchanged.
- Keep the PKL output structure and filenames unchanged.
- Keep policy learning, evaluation, and `amb_radius=0.2` unchanged.
- Update the manuscript equation to show the factor 0.2.
- Do not add per-environment root finding or lookup tables.

## Validation

1. Check the six mean realized KL values using the original 100 replications,
   100 environments, 2,500 test observations, and original seeds.
2. Require an absolute error below \(10^{-4}\) from each requested mean.
3. Run the DGP unit tests and confirm the nominal and concept branches are
   unchanged.
4. Generate the revised covariate-shift data only after these checks pass.

## Consequence

The revised covariate environments differ from the original saved data.
Therefore, the covariate-shift panel of Table 2 must be rerun before its numbers
can be reported; the original covariate-shift results cannot be retained.
