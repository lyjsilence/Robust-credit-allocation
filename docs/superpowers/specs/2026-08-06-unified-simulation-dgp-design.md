# Unified Simulation DGP Design

## Scope

Merge the two DGP modules in `simulation/` into `generate_dgp.py`:

- the original smooth-outcome linear/nonlinear DGP and its nominal,
  covariate-shift, and concept-drift variants;
- the lower-tail-risk DGP and shift evaluators used by the revised Table 2.

The original `code/` directory, policy-learning implementation, evaluation
criterion, generated data, and completed experiment results are out of scope.

## Implementation

Append the lower-tail constants and functions to `generate_dgp.py` without
changing their bodies or the existing smooth-DGP functions. Keep the existing
`generate_dgp.py` command-line interface unchanged. Update the Table 2 runner
and tests to import all DGP functions from `generate_dgp.py`. Delete
`catastrophe_mixture_concept_drift.py` only after validation succeeds.

## Validation

Treat all DGP outputs as deterministic and require exact equality.

1. Before editing, generate fixed-seed in-memory reference outputs for the
   lower-tail functions.
2. After merging, compare every reference scalar and array, including shape,
   dtype, and values.
3. Generate one replication for each reported smooth-DGP family: linear
   nominal, nonlinear nominal, nonlinear covariate shift, and nonlinear
   concept drift. Compare the generated PKL objects recursively with the
   previously verified PKLs.
4. Run the DGP unit tests and Python compilation checks.
5. Run Table 2 replication 0 and require both shift panels to match the
   corresponding rows in the final 100-replication results exactly.

Temporary validation files will be removed after the checks. The final
100-replication results remain unchanged.
