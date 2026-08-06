# Robust Credit Allocation

Code for the simulation and empirical studies of robust credit allocation
under distribution shift.

## Repository structure

- `simulation/`: simulation data generation, policy learning, evaluation, and
  result aggregation code.
- `empirical/`: empirical data preparation, policy learning, and evaluation
  code.

Datasets, trained models, generated results, and reproduction artifacts are
intentionally excluded from this repository.

## Revised concept-drift experiment

From the repository root, reproduce the revised 100-replication Table 2
experiment with:

```bash
python simulation/run_catastrophe_mixture_concept_drift.py \
  --n-replications 100 \
  --output-dir simulation/results/catastrophe_mixture_concept_drift_100rep
```

The runner keeps the training radius at `0.2`, evaluates 100 deployment
environments, and writes both replication-level results and the aggregated
Table 2 CSV.
