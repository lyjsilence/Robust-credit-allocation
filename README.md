# Robust Credit Allocation

Simulation code for the paper *Robust Credit Allocation under Distribution
Shift*. The repository contains the data-generating processes, policy-learning
implementation, evaluation routines, and result aggregation script used for
Tables 1--3.

Generated datasets, trained models, and numerical results are not included.

## Setup

The experiments were run in the Conda environment `yijun_env`:

```bash
conda activate yijun_env
```

Run all commands below from the repository root. The main dependencies are
Python, NumPy, PyTorch, and Joblib. CUDA is used automatically when available;
otherwise, the experiment runner uses the CPU.

## Repository structure

```text
simulation/
  generate_dgp.py     Generate raw, covariate-shift, and concept-drift data
  data.py             Prepare training and validation samples
  models.py           Policy and propensity models
  learning.py         Policy-learning routines
  evaluation.py       Policy-value and worst-environment evaluation
  run_experiment.py   Run one simulation condition
  concat_results.py   Aggregate replication-level results
```

The scripts create the following local directories, all of which are ignored
by Git:

```text
simulation/data/       Generated datasets
simulation/models/     Fitted policy models
simulation/results/    Replication-level results and summary_results.csv
```

## 1. Generate simulation data

Each reported experiment uses 100 replications. `--start_rep` can be used to
split replications across several processes or machines.

### Tables 1 and 3: nominal DGPs

Generate the linear and nonlinear nominal datasets:

```bash
python simulation/generate_dgp.py --mode raw --type linear --dist_shift 0 --num_rep 100
python simulation/generate_dgp.py --mode raw --type nonlinear --dist_shift 0 --num_rep 100
```

These commands save data in `simulation/data/linear/` and
`simulation/data/nonlinear/`.

### Table 2: covariate shift and concept drift

For each mode, repeat the command at
`LEVEL = 0, 0.5, 1, 1.5, 2, 2.5, 3`:

```bash
python simulation/generate_dgp.py --mode shift --type nonlinear --dist_shift LEVEL --num_rep 100
python simulation/generate_dgp.py --mode drift --type nonlinear --dist_shift LEVEL --num_rep 100
```

For example, the covariate-shift command at level 1.5 is:

```bash
python simulation/generate_dgp.py --mode shift --type nonlinear --dist_shift 1.5 --num_rep 100
```

The zero-severity shift and drift conditions are intentionally distinct from
the nominal nonlinear condition. Their directories are respectively
`nonlinear_shift=0.0`, `nonlinear_drift=0.0`, and `nonlinear`.

## 2. Run the experiments

The default training ambiguity radius is 0.20. Every command below runs
replications 0--99 and writes models and results into matching condition
directories.

### Table 1: linear and nonlinear policies

Run the proposed method and the SNLS baseline for each nominal DGP:

```bash
python simulation/run_experiment.py --mode raw --type linear --dist_shift 0 --num_reps 100 -robust_met robust -baseline None
python simulation/run_experiment.py --mode raw --type linear --dist_shift 0 --num_reps 100 -robust_met robust -baseline SNLN

python simulation/run_experiment.py --mode raw --type nonlinear --dist_shift 0 --num_reps 100 -robust_met robust -baseline None
python simulation/run_experiment.py --mode raw --type nonlinear --dist_shift 0 --num_reps 100 -robust_met robust -baseline SNLN
```

The KNLS baseline is implemented as `-baseline KLLN` but is not required for
the reported SNLS-only comparison.

### Table 2: robustness under distribution shift

For every `MODE` in `shift, drift` and every `LEVEL` in
`0, 0.5, 1, 1.5, 2, 2.5, 3`, run both policies:

```bash
python simulation/run_experiment.py --mode MODE --type nonlinear --dist_shift LEVEL --num_reps 100 -robust_met robust -baseline None
python simulation/run_experiment.py --mode MODE --type nonlinear --dist_shift LEVEL --num_reps 100 -robust_met non-robust -baseline None
```

For example:

```bash
python simulation/run_experiment.py --mode drift --type nonlinear --dist_shift 1.5 --num_reps 100 -robust_met robust -baseline None
python simulation/run_experiment.py --mode drift --type nonlinear --dist_shift 1.5 --num_reps 100 -robust_met non-robust -baseline None
```

### Table 3: policy complexity

No additional experiment command is needed. The nonlinear proposed-method run
for Table 1 automatically fits and saves fixed complexities
`K = 1, ..., 6`, together with the data-driven selection. Therefore, Table 3
uses the results produced by:

```bash
python simulation/run_experiment.py --mode raw --type nonlinear --dist_shift 0 --num_reps 100 -robust_met robust -baseline None
```

## 3. Aggregate results

After the required experiments finish, aggregate every existing result file:

```bash
python simulation/concat_results.py --results_dir simulation/results
```

The script writes:

```text
simulation/results/summary_results.csv
```

Each row reports the DGP type, perturbation mode and severity, robust method,
baseline, metric name, number of completed replications, mean, sample standard
deviation, and standard error. The same summary contains the entries needed for
Tables 1, 2, and 3, including all fixed-`K` and data-driven nonlinear results.

To combine only a subset of replications, place the corresponding `.pkl` files
under the standard results directory and run the same aggregation command.
