# Distinct Shift and Drift Summary Names

## Goal

Prevent covariate-shift and concept-drift summaries with the same severity from
overwriting each other.

## Output Names

- Nominal: `summary_Nominal.csv`
- Nominal baseline: `summary_Nominal-SNLN.csv` or
  `summary_Nominal-KLLN.csv`
- Covariate shift: `summary_Shift_<level>.csv`
- Concept drift: `summary_Drift_<level>.csv`

Integer-valued levels continue to omit `.0`, so examples include
`summary_Shift_1.csv` and `summary_Drift_1.5.csv`.

## Scope

- Change only the summary output filename in `simulation/concat_results.py`.
- Keep result-file discovery, completeness checks, statistics, CSV columns, and
  command-line arguments unchanged.
- Keep robust and non-robust summaries in their existing separate directories.
- Do not delete or rename existing `summary_Dist_*.csv` files.

## Validation

1. Run syntax checking in `yijun_env`.
2. Check filename generation for nominal, shift, drift, and baseline cases.
3. Run available shift and drift summaries at the same severity and confirm
   they produce different paths without overwriting each other.
