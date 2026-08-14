# Action-Specific Unified Simulation Sandbox

## Scope

Create an isolated sandbox for all simulation exhibits. The formal `simulation/data`,
`simulation/models`, and `simulation/results` directories remain unchanged. Run 100
replications for Tables 1--3, and produce the decision-boundary figures. Only the
SNLS linear baseline is included; KNLS is excluded.

## Master DGP

Draw five covariates uniformly from the unit ball. Let `s(X1, X2)` be either a
linear score or the paper's S-shaped nonlinear score. For action `a`, define

```text
logit p_a(X) = alpha + rho v(X) - (2a - 1) delta tanh(gamma s(X1, X2)),
Y(a) = y_good if U_a > p_a(X), and y_bad otherwise.
```

The common tail-risk term `v(X)` changes outcome levels without changing the
nominal optimal boundary: action 1 is optimal exactly when `s(X1, X2) > 0`.
The logging policy selects the nominally optimal action with probability 0.75.

- Raw: keep both `P(X)` and `P(Y | X, A)` fixed.
- Covariate shift: exponentially tilt `P(X)` toward high tail-risk observations;
  keep `P(Y | X, A)` fixed.
- Concept drift: keep `P(X)` fixed and increase action-specific catastrophe
  logits using heterogeneous susceptibilities.

Both shifts use KL targets 0.005, 0.02, 0.045, 0.08, 0.125, and 0.18.

## Experiments

1. Table 1: train at radius 0.20 and evaluate at radii 0.05, 0.10, 0.20,
   0.30, and 0.40 for linear and nonlinear scores. Compare Ours with SNLS.
2. Table 2: on the nonlinear DGP, compare the same data-driven robust learner
   with its non-robust counterpart under both shift modes.
3. Table 3: compare fixed depths K=1,...,6 with the holdout-penalized selected
   depth. Candidate policies are fitted once and reused for the fixed-depth and
   selected-depth rows.
4. Figures: plot the true boundary and learned Ours/SNLS decisions for the
   linear and nonlinear replications.

## Execution and Reporting

Use the existing policy learner and evaluation objective without changing their
mathematics. Run a small pilot first. Parameter changes prompted by the pilot are
recorded in `search_log.csv`; the selected specification is then frozen before
the 100-replication run. Report means and Monte Carlo standard errors, selected
depth frequencies, exact realized KL errors, output completeness, and the random
seed convention.

Generated datasets, model checkpoints, raw result files, and logs are ignored by
Git. Summary tables, figures, code, the frozen configuration, and the search log
remain tracked.
