# Automatic Summary of Existing Simulation Results

## Goal

Update `simulation/concat_results.py` so one command summarizes every existing
simulation result under `simulation/results`. The script should be easy to read
and should not require the user to run it separately for each DGP condition.

## Input discovery

The script will recursively find all `.pkl` files below the results directory.
It will infer the experiment metadata from the existing directory and filename
conventions:

- DGP type: `linear` or `nonlinear`
- mode: `raw`, `shift`, or `drift`
- severity: `0` for raw, otherwise the value in the condition directory
- method: `robust` or `non-robust`
- baseline: `None`, `SNLN`, or `KLLN`
- replication ID: the value after `exp_id-`

The default results directory will be resolved relative to
`concat_results.py`, so the command works from either the repository root or
the `simulation` directory.

## Aggregation and output

Each key stored in a result file is treated as one metric. Results are grouped
by DGP type, mode, severity, method, baseline, and metric. The script uses all
replications that currently exist instead of requiring a fixed number.

One long-format CSV will be written to
`simulation/results/summary_results.csv` with these columns:

```text
dgp_type, mode, severity, method, baseline, metric, n, mean, std
```

The standard deviation is the sample standard deviation when a group has more
than one observation and zero when it has one. Rows will be sorted in a stable,
human-readable order: DGP type, raw/shift/drift, severity, method, baseline,
and metric.

## Code structure

The implementation will use a small set of single-purpose functions:

1. parse metadata for one result path;
2. load metric values from all discovered files;
3. aggregate observations;
4. write the combined CSV.

Only the existing file conventions will be supported. The script will avoid a
large configuration matrix or additional compatibility layers.

## Validation

Run the revised script against the current results tree and verify:

- every existing experimental directory is represented;
- group sizes match the number of available replications, including the
  incomplete linear raw group;
- selected means and standard deviations match the current single-condition
  aggregation logic;
- running from both the repository root and the simulation directory resolves
  the same input and output locations.
