# Fair policy optimization and Table 2 rerun

## Scope

Fix the simulation policy optimizer without changing any DGP, policy class,
training radius, random seed, sample size, or evaluation definition. Rerun the
two Table 2 panels for six shift levels and 100 replications per level.

## Training design

- The non-robust learner selects checkpoints with validation SNIPS/IPW value.
  It trains for at least 100 epochs before early stopping so that an initially
  constant hard policy cannot stop optimization before crossing a decision
  boundary.
- The robust learner selects checkpoints with validation KL-DRO value.
- Both validation criteria are maximized. Early stopping occurs after ten
  consecutive epochs without an improvement of at least `0.001`.
- Each policy-optimization call restores the best validation checkpoint rather
  than the final epoch.
- The robust policy keeps its existing policy-alpha alternating optimization.
  The non-robust policy uses one policy-optimization call because it has no
  alpha block to alternate.

## Validation gates

1. Unit tests verify criterion selection, improvement direction, patience, and
   best-checkpoint restoration.
2. A 10-replication covariate-shift check compares nominal and shifted true
   policy values. The previous large nominal gap must disappear; otherwise the
   full run is stopped for diagnosis.
3. The formal run uses the existing saved PKL data, `amb_radius=0.2`, six
   severities, 100 replications, and both robust methods.
4. Every formal condition must contain exactly 100 result files, with no
   missing or non-finite policy values. Summary statistics are regenerated only
   after this completeness check.

## Output handling

Existing Table 2 result PKLs and summaries may be replaced after the sanity
check passes. DGP PKLs are retained unchanged. Parallel workers use disjoint
replication ranges and shared formal output paths whose filenames do not
overlap. Model checkpoint names include the shift mode so covariate-shift and
concept-drift workers cannot overwrite one another.
