# Robust Credit Allocation

Code for the simulation and empirical studies of robust credit allocation
under distribution shift.

## Repository structure

- `simulation/generate_dgp.py`: all simulation DGPs and deployment-shift
  mechanisms.
- `simulation/`: policy learning, evaluation, and result aggregation code.
- `empirical/`: empirical data preparation, policy learning, and evaluation
  code.

Datasets, trained models, generated results, and reproduction artifacts are
intentionally excluded from this repository.

## Revised Table 2 experiment

From the repository root, reproduce the revised 100-replication Table 2
experiment with:

```bash
python simulation/run_catastrophe_mixture_concept_drift.py \
  --n-replications 100 \
  --output-dir simulation/results/catastrophe_mixture_table2_100rep
```

The runner keeps the training radius at `0.2`, evaluates 100 deployment
environments per shift level, and writes the covariate-shift and concept-drift
panels of Table 2.
