# Restore the Simulation Data Pipeline

## Goal

Restore the journal code to one workflow:

```text
generate_dgp.py -> saved PKL files -> run_experiment.py
```

The original covariate-shift experiment remains unchanged. Only the reported
concept-drift DGP is replaced by the lower-tail-risk specification.

## Code Changes

- Keep the nominal and covariate branches of `generate_dgp.py` unchanged.
- Replace its old smooth structural concept branch with the lower-tail concept
  drift.
- Preserve the existing PKL contract: `train`, `test.base`, and
  `test.envs`.
- Delete `run_catastrophe_mixture_concept_drift.py`.
- Delete the lower-tail covariate evaluator and the old smooth concept-drift
  functions after they have no callers.
- Use `run_experiment.py` for both robust and non-robust policy learning and
  worst-environment evaluation.
- Preserve the user's current uncommitted simplifications in
  `generate_dgp.py`.

## Results and Writing

- Retain the original covariate-shift data and Table 2 results.
- Retain only the new lower-tail concept-drift results.
- Revise Sections 5.1 and 5.3 so the two panels share a general simulation
  framework but use different conditional outcome specifications.

## Validation

Generate replication 0 for all six reported concept severities. Check the PKL
structure, run the existing `run_experiment.py` once for robust and once for
non-robust at every severity, and compare the resulting \(Q_{\min}\) values
with replication 0 of the previously verified concept-drift result. Exact
equality is required. The original covariate-shift files are not regenerated.
