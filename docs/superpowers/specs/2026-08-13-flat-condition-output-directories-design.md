# Flat Condition Output Directories

## Goal

Remove the redundant `Simulation` level from simulation outputs and organize
results by the same DGP condition used for data and models.

## Layout

From the `simulation/` working directory:

- raw: `results/<type>/<robust_met>/` and
  `models/<type>/<robust_met>/`;
- shift: `results/nonlinear_shift=<severity>/<robust_met>/` and
  `models/nonlinear_shift=<severity>/<robust_met>/`;
- drift: `results/nonlinear_drift=<severity>/<robust_met>/` and
  `models/nonlinear_drift=<severity>/<robust_met>/`.

`run_experiment.py` will compute the condition once and use it for both output
roots. `concat_results.py` will use the same mapping. Model filenames, result
filenames, training, evaluation, and random seeds remain unchanged.

Existing `results/Simulation/` and `models/Simulation/` directories will not
be moved or deleted.

## Validation

Verify all raw, shift, and drift paths statically, then run one isolated
replication for raw, shift=0, and drift=0. Confirm distinct result and model
directories and successful aggregation without writing to the old
`Simulation` subtree.
