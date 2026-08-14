# Autonomous Covariate-Shift DGP Search

## Authorization and scientific status

The user authorized an autonomous search for a covariate-shift DGP whose robust
advantage increases materially with shift severity. No intermediate approval is
required. The search is exploratory: every attempted configuration and result
must be logged, and a selected DGP must not later be described as pre-specified
confirmatory evidence.

All work is isolated below `simulation/shift_dgp_search`. Existing simulation
code, data, models, and results are read-only.

## Fixed experiment settings

- Policy learning and evaluation: import the current unchanged core functions.
- Training size: 10,000; test size: 2,500; 100 test environments.
- Ambiguity radius: 0.2; all other learning settings match Table 2.
- Severities: 0.5, 1, 1.5, 2, 2.5, 3.
- Target KL: 0.005, 0.02, 0.045, 0.08, 0.125, 0.18.
- Full success: paired mean improvement is nondecreasing, reaches at least 10%
  at severity 3, and rises by at least 8 percentage points from severity 0.5.

## Candidate family: structured risk islands

Covariates remain uniform on the five-dimensional unit ball. The nonlinear
score in `(X1, X2)` selects which action is risky. A deliberately structured,
high-frequency vulnerability surface uses only observed covariates:

```text
u_k(X) = sin(k pi X3) sin(k pi X4)
v(X)   = sigmoid(sharpness * (u_k(X) - threshold)).
```

The stable action is deterministic. The risky action equals `good_outcome` or
zero, with catastrophe probability

```text
p(X) = p_low + (p_high - p_low) v(X).
```

This conditional outcome mechanism is fixed in training and test data. Test
weights are `q_lambda(X) proportional to exp(beta_lambda v(X))`, with beta
calibrated to the target KL. Thus only `P(X)` changes.

The initial grid varies frequency (`k=2,4,6`), threshold (`0.10,0.25`), and a
small set of fixed reward/risk profiles. The grid does not use learned policy
outputs. If needed, a second grid may vary logging overlap, but it must retain
known propensities and be logged separately.

## Search stages

1. Generate and run three paired replications at severities 0.5, 1.5, and 3 for
   each initial candidate.
2. Rank candidates by monotonicity, severity-3 improvement, and total increase.
3. Run ten paired replications at all six severities for promising candidates.
4. Run 100 paired replications only for a candidate that passes the ten-run
   gate: nondecreasing means, at least 8% at severity 3, and at least a
   6-percentage-point increase.
5. Apply the fixed full success criterion to the 100-run result.

The search may continue within the stated DGP family if a candidate fails, but
every attempt stays in `attempts.csv`. No change to policy learning, evaluation,
the improvement definition, or KL targets is allowed.

## Files

```text
simulation/shift_dgp_search/
  dgp.py
  run_experiment.py
  summarize.py
  attempts.csv
  data/
  models/
  results/
```

Generated artifacts are ignored by Git. Once a candidate succeeds, retain only
the readable selected DGP configuration, the scripts needed to reproduce it,
the 100-run summary, and the complete attempt log. Raw generated data and models
remain local and are not uploaded.

## Verification

- Check every realized KL against its target within numerical tolerance.
- Use the same learned policy across severities within each method/replication,
  because training data do not depend on deployment severity.
- Compute paired improvement and 95% confidence intervals.
- Confirm formal simulation directories and user-owned working-tree changes are
  untouched.
