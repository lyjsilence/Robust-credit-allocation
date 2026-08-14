# Unified tail-risk sandbox design

## Objective

Test covariate shift and concept drift within one probabilistic tail-risk
framework while leaving the formal simulation code and the existing isolated
search results unchanged.

## Isolation

All new scripts and generated artifacts live under
`simulation/shift_dgp_search/unified_tail_risk`. The formal `simulation/data`,
`simulation/models`, and `simulation/results` directories are never read as
output targets. The previously selected `islands_k2_t10` data and results are
also left unchanged.

## Shared outcome framework

Covariates are uniform on the five-dimensional unit ball. The existing
nonlinear boundary in `(X1, X2)` identifies a risky action; the other action is
stable. Both shift modes use:

- stable outcome: 0.79;
- good risky outcome: 0.90;
- bad risky outcome: 0;
- logging probability of the risky action: 0.75.

The risky outcome is determined by a Bernoulli catastrophe. Thus both modes
share the same outcome family and scale.

## Covariate shift

The baseline catastrophe probability is a fixed function of a smooth risk score
based on `(X3, X4)`. Its population average is approximately 0.10, but it crosses
the 0.1222 threshold above which the stable action has the larger conditional
mean. Deployment covariates are exponentially tilted toward high-risk regions.
Only `P(X)` changes; `P(Y | X,T)` remains fixed. Tilt parameters are calibrated
to KL values 0.005, 0.02, 0.045, 0.08, 0.125, and 0.18.

Candidate risk-score thresholds are screened analytically before policy runs.
The selected candidate must have approximately 0.10 nominal average catastrophe
risk, a small method difference at lambda 0.5, and a materially increasing
high-lambda robust advantage.

## Concept drift

The baseline catastrophe probability is 0.10, matching the current formal
concept-drift design. The covariate distribution remains fixed. Each deployment
environment changes conditional catastrophe probabilities through a positive,
X-dependent susceptibility on the logit scale. The average conditional
Bernoulli KL is calibrated to the same six target values. Common catastrophe
uniforms are reused across environments.

## Learning and evaluation

The sandbox imports the formal policy-learning and policy-evaluation functions
without modifying them. The ambiguity radius remains 0.2. Each replication
learns one policy per method and evaluates that policy at all six severities.

## Execution gates

1. Verify schemas, target KL values, and deterministic regeneration.
2. Run a short pilot for both modes.
3. If the covariate-shift improvement rises materially and concept drift remains
   coherent, run 100 replications for both modes.
4. Report method means, improvement of means, paired mean improvement, and 95%
   confidence intervals.

The final result is exploratory because the covariate DGP is selected after
candidate screening.
