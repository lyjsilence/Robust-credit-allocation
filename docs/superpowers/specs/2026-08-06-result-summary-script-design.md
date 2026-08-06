# Simulation Result Summary Script Design

## Goal

Update `simulation/concat_results.py` so it summarizes the `.pkl` files
produced by the current `run_experiment.py` without changing the experiment
runner or result files.

## Interface

Keep the existing command-line arguments:

- `--base_dir`
- `--mode`
- `--robust_met`
- `--type`
- `--baseline`
- `--dist_shift`
- `--num_runs`
- `--out_root`

`--dist_shift` accepts any float, including the reported levels
`0, 0.5, 1, 1.5, 2, 2.5, 3`.

## File Resolution

Use the same severity-label rule as `run_experiment.py`: integer-valued
severities omit `.0`.

- At severity zero, read `Shift_0.0-...-DRO.pkl`.
- Under nonzero covariate shift, read `Shift_<level>-...-min.pkl`.
- Under nonzero concept drift, read `Drift_<level>-...-min.pkl`.
- Preserve the existing `SNLN` and `KLLN` filename variants.

## Output

Write one CSV for the selected method and condition:

```text
summary_Dist_<level>.csv
```

or the corresponding `SNLN`/`KLLN` filename. Each row contains:

```text
key,n,mean,std
```

The standard deviation is the sample standard deviation (`ddof=1`).
Robust and non-robust results remain separate invocations. The script does not
calculate the Table 2 improvement percentage.

## Completeness Check

Before aggregating, require all replication IDs from zero through
`num_runs - 1`. If any file is absent, stop and report the missing IDs rather
than producing a partial summary.

## Cleanup

Remove duplicated wrapper functions, unused imports and arguments, and corrupted
comments while keeping the script short and readable.

## Validation

1. Run syntax checking in `yijun_env`.
2. Summarize the existing 100 nonlinear robust nominal results.
3. Verify that the CSV contains the expected metrics, `n=100`, and independently
   recomputed means and sample standard deviations.
4. Confirm that an incomplete result directory reports missing replication IDs.
