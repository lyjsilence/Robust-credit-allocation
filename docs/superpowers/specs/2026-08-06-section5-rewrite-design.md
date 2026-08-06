# Section 5.1 and 5.3 Rewrite Design

## Scope

Rewrite Sections 5.1 and 5.3 without changing the experiments reported in
Sections 5.2 or 5.4. The revision will define one general simulation framework
and present the smooth-outcome and lower-tail-risk models as two transparent
specifications within that framework.

## Section 5.1

Define the simulation through three components:

1. a covariate distribution `P_X`;
2. a score-induced preferred action and behavior policy;
3. a conditional outcome kernel `P_vartheta(Y | X, A)`.

The existing Gaussian-sigmoid model remains the smooth-outcome specification
used for the policy-class and complexity experiments. The lower-tail-risk
model is the specification used for the robust-versus-non-robust comparison in
Section 5.3.

For the lower-tail model:

```text
p_0 = 0.10
Y_risky = 0.00 with probability p_0, and 0.90 otherwise
Y_stable = 0.79
```

The behavior policy selects the risky action with probability `0.75`.

## Section 5.3

Both Table 2 panels use the same lower-tail baseline DGP and the same learned
policies.

- Covariate shift changes `P_X` by the existing exponential-tilting and
  resampling mechanism while holding the conditional outcome kernel fixed.
- Concept drift holds `P_X` fixed and changes only the risky action's
  catastrophe probability. The probability tilt is calibrated so its mean
  conditional Bernoulli KL equals
  `0.5 * (0.2 * lambda_2)^2`.

The score function, action boundary, behavior policy, outcome support,
training sample, and policy-learning configuration remain fixed within both
panels.

## Experiment update

Extend the final Table 2 runner so that robust and non-robust policies are
trained once per replication and evaluated under both shifts. Run 100
replications and 100 environments at each severity.

The new covariate panel replaces the old covariate numbers in Table 2. The
concept panel must reproduce the already verified concept-drift results.
Table 1, Table 3, Figures 3-4, and their underlying simulations are unchanged.

## Writing output

Write a standalone English replacement draft containing:

- complete revised Section 5.1;
- complete revised Section 5.3;
- the full two-panel Table 2;
- a short revision note explaining what changed.

Do not modify the proof PDF. Save the draft as Markdown under
`docs/manuscript/`.

## Validation

- Every reported Table 2 cell must be read from the 100-replication output.
- Each panel-severity combination must contain exactly 100 rows.
- The concept panel must match the prior verified output.
- No claims of cross-panel comparability are needed because both panels share
  the same baseline DGP and policies.
- The prose must preserve the paper's notation and formal
  operations-research/computing register.
