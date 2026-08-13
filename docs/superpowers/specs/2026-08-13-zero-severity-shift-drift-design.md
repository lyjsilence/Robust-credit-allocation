# Zero-Severity Shift and Drift Conditions

## Goal

Treat `raw`, `shift=0`, and `drift=0` as three distinct simulation conditions.
They use different DGPs and saved-data contracts even though all have zero
deployment severity.

## Behavior

- `raw` reads `data/<type>/Shift_0.0-exp-<rep>.pkl` and uses nominal DRO
  evaluation.
- `shift=0` reads
  `data/nonlinear_shift=0.0/Shift_0-exp-<rep>.pkl` and uses minimum-environment
  evaluation.
- `drift=0` reads
  `data/nonlinear_drift=0.0/Drift_0-exp-<rep>.pkl` and uses
  minimum-environment evaluation.

Model directories mirror these three data conditions. Result files remain in
the existing result directory but use distinct names:

- raw: `Shift_0.0-...-DRO.pkl`;
- shift=0: `Shift_0-...-min.pkl`;
- drift=0: `Drift_0-...-min.pkl`.

`basics_with_holdout.py` will use `mode`, rather than the numeric severity, to
select the raw versus environment-list data contract. `concat_results.py` will
use the same mode-based filename distinction.

## Validation

Check all three path mappings and run one isolated replication each for raw,
shift=0, and drift=0. Confirm that each writes to its distinct model directory
and result filename, while existing nonzero shift/drift naming remains
unchanged.
