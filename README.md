# Robust Credit Allocation

Code for the simulation and empirical studies of robust credit allocation
under distribution shift.

## Repository structure

- `simulation/generate_dgp.py`: the unified DGP used for the revised simulation
  tables and figures.
- `simulation/`: policy learning, evaluation, and result aggregation code.
- `empirical/`: empirical data preparation, policy learning, and evaluation
  code.

Datasets, trained models, generated results, and reproduction artifacts are
intentionally excluded from this repository.

## Simulation workflow

Generate and save the simulation data before running policy learning. Select
the DGP explicitly with `--mode raw`, `--mode shift`, or `--mode drift`. For
example, generate the unified concept-drift data at \(\lambda_2=1\):

```bash
python simulation/generate_dgp.py \
  --mode drift \
  --dist_shift 1 \
  --num_rep 100 \
  --out_dir simulation/data
```

Then run the existing experiment entry point:

```bash
python simulation/run_experiment.py \
  --mode drift \
  --dist_shift 1 \
  -robust_met robust
```

`generate_dgp.py` saves each condition in the directory read by
`run_experiment.py`, for example `simulation/data/nonlinear_drift=1.0` or
`simulation/data/nonlinear_shift=0.5`. Repeat the last command with
`-robust_met non-robust`.

The three zero-severity conditions remain distinct:

```bash
python simulation/generate_dgp.py --mode raw --type nonlinear --dist_shift 0
python simulation/generate_dgp.py --mode shift --dist_shift 0
python simulation/generate_dgp.py --mode drift --dist_shift 0
```

They read `data/nonlinear`, `data/nonlinear_shift=0.0`, and
`data/nonlinear_drift=0.0`, respectively. Results and models mirror the same
condition directories directly under `results/` and `models/`.
