# Mirror Model Condition Directories

## Goal

Store simulation models under the same condition directories as generated
data, without changing training, evaluation, model filenames, or numerical
results.

## Directory mapping

- `raw`: `models/Simulation/<type>/<robust_met>/`
- `shift`: `models/Simulation/nonlinear_shift=<severity>/<robust_met>/`
- `drift`: `models/Simulation/nonlinear_drift=<severity>/<robust_met>/`

`run_experiment.py` will compute the condition name once and use it for both
the data directory and model directory. It will pass the resolved model
directory to `opt_policy_learn_with_holdout.py`, which will continue to use
the existing model filename.

Results remain under `results/Simulation/<type>/<robust_met>/` so existing
aggregation code and Table 2 files are unaffected.

## Validation

Run the same replication with the pre-change and post-change runners using
identical data, seeds, and settings. Compare every saved model tensor and
every result value exactly, and confirm only the model directory differs.
