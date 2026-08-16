# Robust Credit Allocation

Simulation code for the paper *Policy Learning under Distribution Shift with
Infinite-Dimensional Policy Classes: An Application
to Consumer Credit Allocation*. The repository contains the data-generating processes, policy-learning
implementation, evaluation routines, and result aggregation script.


## Requirements

The simulation code was tested with the following versions:

- Python 3.8.20
- NumPy 1.24.1
- Joblib 1.4.2
- PyTorch 2.3.0+cu118
- CUDA 11.8

Install the required Python packages with:

```bash
python -m pip install -r requirements.txt
```

The CUDA build is only required for GPU execution. PyTorch automatically uses
the CPU when CUDA is unavailable.

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

The scripts create the following local directories:

```text
simulation/data/       Generated datasets
simulation/models/     Fitted policy models
simulation/results/    Replication-level results and summary_results.csv
```

## 1. Generate simulation data

Each reported experiment uses 100 replications. `--start_rep` can be used to
split replications across several processes or machines.

### Generate nominal DGPs

Generate the linear and nonlinear nominal datasets:

```bash
python generate_dgp.py --mode raw --type linear --dist_shift 0 --num_rep 100
python generate_dgp.py --mode raw --type nonlinear --dist_shift 0 --num_rep 100
```

### Generate covariate shift and concept drift

For each mode, repeat the command at
`LEVEL = 0, 0.5, 1, 1.5, 2, 2.5, 3`:

```bash
python generate_dgp.py --mode shift --type nonlinear --dist_shift LEVEL --num_rep 100
python generate_dgp.py --mode drift --type nonlinear --dist_shift LEVEL --num_rep 100
```

## 2. Run the experiments

The default training ambiguity radius is 0.20. Every command below runs
replications 0--99 and writes models and results into matching condition
directories.

### (1) linear and nonlinear policies

Run the proposed method for each nominal DGP:

```bash
python simulation/run_experiment.py --mode raw --type linear --dist_shift 0 --num_reps 100 -robust_met robust
python simulation/run_experiment.py --mode raw --type nonlinear --dist_shift 0 --num_reps 100 -robust_met robust 
```


### (2) robustness under distribution shift

For every `MODE` in `shift, drift` and every `LEVEL` in
`0, 0.5, 1, 1.5, 2, 2.5, 3`, run both policies:

```bash
python simulation/run_experiment.py --mode MODE --type nonlinear --dist_shift LEVEL --num_reps 100 -robust_met robust 
python simulation/run_experiment.py --mode MODE --type nonlinear --dist_shift LEVEL --num_reps 100 -robust_met non-robust 
```

### (3) policy complexity

The nonlinear proposed-method run
for Table 1 automatically fits and saves fixed complexities
`K = 1, ..., 6`, together with the data-driven selection. Therefore, Table 3
uses the results produced by:

```bash
python simulation/run_experiment.py --mode raw --type nonlinear --dist_shift 0 --num_reps 100 -robust_met robust
```

## 3. Aggregate results

After the required experiments finish, aggregate every existing result file:

```bash
python sconcat_results.py --results_dir results
```

The script writes:

```text
simulation/results/summary_results.csv
```

