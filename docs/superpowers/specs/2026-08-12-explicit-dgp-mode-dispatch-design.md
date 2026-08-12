# Explicit DGP Mode Dispatch

## Goal

Make the three simulation choices immediately visible in
`simulation/generate_dgp.py` without changing generated data, random-number
order, saved file names, or the experiment interface.

## Design

The public command is `--mode raw|shift|drift`. Each mode has one named
generation function, while `generate_replication` is a short dispatcher:

- `raw`: smooth-outcome training and unshifted test data;
- `shift`: the same smooth outcomes with X-dependent covariate resampling;
- `drift`: the lower-tail-risk outcome model with concept drift.

The historical command names `none`, `covariate`, and `concept` are retained
as hidden `--shift_type` aliases so existing scripts continue to work. Output
directories and PKL schemas remain unchanged.

## Validation

Run the simulation unit tests, exercise both the new and legacy command-line
interfaces, and compare newly generated representative raw, shift, and drift
objects with the existing PKL files recursively and exactly.
