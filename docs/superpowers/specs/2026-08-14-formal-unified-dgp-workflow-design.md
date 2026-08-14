# Formal Unified-DGP Workflow

## Goal

Move the verified unified DGP out of the exploratory sandbox so that datasets
can be generated explicitly from command-line arguments and consumed by the
formal `simulation/run_experiment.py` workflow.

## Generator

Create `simulation/generate_unified_dgp.py` as one self-contained file. It will
accept `raw`, `shift`, and `drift` modes; linear/nonlinear boundary type;
severity; replication range; sample sizes; and number of environments. Outputs
will use the existing condition directories and filenames under
`simulation/data/`.

Each file will contain top-level `train` and `test` datasets. Shift and drift
files will additionally contain a `specification` object holding, respectively,
the environment resampling indices or drift probabilities and realized KL.

## Experiment runner

Keep the existing command-line interface, model/result directories, and output
filenames. Adapt only the orchestration needed to reproduce the verified run:

- deterministic replication- and model-specific seeds;
- candidate depths 1--6 for nonlinear Ours, 1--2 for linear Ours and Table 2,
  and fixed depth 2 for SNLS;
- direct worst-environment evaluation from shift indices or drift
  probabilities;
- the existing DRO evaluation for raw data.

The policy network, optimization objective, training functions, and DRO
evaluator remain unchanged.

## Verification

1. Compare formal-generator replication 0 with the sandbox arrays exactly.
2. Run one representative raw, shift, and drift experiment.
3. Compare the formal results with calculations from the saved sandbox models
   where the same fitted-policy specification is requested.
4. Confirm no changes to the four core learning/evaluation modules.
5. Do not overwrite the existing formal or sandbox outputs during testing.
