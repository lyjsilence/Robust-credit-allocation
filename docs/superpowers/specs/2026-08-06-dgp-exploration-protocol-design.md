# Concept-Drift DGP Exploration Protocol

## Objective

Find a transparent concept-drift DGP under which the existing Table 2 robust
policy has an increasing relative advantage as `lambda_2` increases. The
existing policy learner, training radius (`0.2`), sample sizes, six severity
levels, 100 deployment environments, and 100-replication final design remain
fixed.

The exploration must not alter the robust estimator or select a final result
from the same random seeds used for reporting.

## Candidate approaches

Three approaches were considered:

1. Favorable drift for the action selected more often by the robust policy.
   This is easy to engineer but gives a weak robustness interpretation.
2. Search transparent lower-tail DGP families for a setting in which the
   existing robust objective selects the risky action less often than the
   non-robust objective. This is the selected approach.
3. Construct drift after observing policy disagreements. This would directly
   target the fitted policies and is excluded as unsuitable for a journal
   experiment.

## DGP search space

All candidates use the original five-dimensional uniform-ball covariates and
nonlinear boundary. The boundary defines the high-return action. Candidate
families vary only a small number of interpretable outcome parameters:

- baseline catastrophe probability;
- catastrophe and non-catastrophe outcomes;
- stable-action outcome;
- logging probability for the high-return action;
- whether catastrophe risk is constant or smoothly heterogeneous in distance
  from the nonlinear boundary.

The high-return action must have a higher nominal population mean. The stable
action must have a higher population KL worst-case value at radius `0.2`.
Candidates that fail either analytic condition are discarded before policy
training.

Deployment drift increases only the high-return action's catastrophe
probability. A scalar tilt is solved separately for every environment and
severity so the mean conditional Bernoulli KL is exactly

```text
eta_test = 0.5 * (0.2 * lambda_2)^2.
```

Common uniforms make catastrophes nested across severity.

## Search and validation split

The process has three stages:

1. **Discovery:** use replication IDs `1000-1002` to compare analytically
   screened candidates. These results may select a candidate.
2. **Holdout validation:** freeze the selected DGP and test it on replications
   `2000-2009`. If it fails, return to discovery and record the failure; do not
   relabel the holdout as discovery.
3. **Final run:** after one candidate passes holdout validation, run the fixed
   DGP for replication IDs `0-99`.

Temporary discovery scripts, models, and result directories are not part of
the journal code and will be removed after the final candidate is fixed.

## Acceptance criteria

A candidate enters holdout validation only if its discovery aggregate has:

- non-robust risky-action exposure at least five percentage points above
  robust exposure;
- positive robust-minus-non-robust value at the largest severity;
- larger relative improvement at `lambda_2 = 3` than at `lambda_2 = 0.5`;
- Spearman association of at least `0.8` between severity and relative
  improvement.

The frozen candidate passes holdout validation only if the same four
conditions hold on the aggregate of the ten unused replications. The six-point
aggregate must have at least four nonnegative adjacent changes. The one allowed
decrease must be smaller than `0.25` percentage points.

The final 100-replication table is reported regardless of whether every point
remains monotone. No parameter changes are allowed after the holdout gate.

## Final artifacts

Keep only:

- one concise DGP module;
- one Table 2 runner;
- focused mechanism tests;
- replication-level and Table 2 aggregate CSV files;
- an audit log of all candidate families and gates.

Do not modify the original paper code or historical simulation data.
