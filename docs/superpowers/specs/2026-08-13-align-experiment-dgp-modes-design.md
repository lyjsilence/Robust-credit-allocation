# Align Experiment DGP Modes

## Goal

Use the same `raw`, `shift`, and `drift` mode names in `run_experiment.py` as
in `generate_dgp.py`, without changing policy learning, evaluation, random
seeds, model files, or result files.

## Changes

- Expose one command-line option: `--mode raw|shift|drift`, defaulting to
  `drift`.
- Keep the existing data directories:
  - `raw` reads `data/<type>/Shift_0.0-exp-<rep>.pkl`;
  - `shift` reads
    `data/nonlinear_shift=<severity>/Shift_<severity>-exp-<rep>.pkl`;
  - `drift` reads
    `data/nonlinear_drift=<severity>/Drift_<severity>-exp-<rep>.pkl`.
- Keep nominal result names beginning with `Shift_0`; keep nonzero shift and
  drift result names unchanged so `concat_results.py` remains compatible.
- Do not alter training or evaluation functions.

## Validation

- Unit-test directory and filename resolution for all three modes.
- Run the existing simulation tests in `yijun_env`.
- Compare representative pre-change and post-change experiment outputs using
  the same generated PKL and seeds. Values must be exactly equal.
