# Unified simulation sandbox

This isolated directory runs Tables 1--3 and the linear/S-shaped decision-boundary
figures without changing the formal simulation folders.

The DGP family shares the five-dimensional unit-ball covariates, the linear or
S-shaped score `s(X1, X2)`, the action mapping, logging propensity, and KL
calibration. Tables 1 and 3 use the smooth outcome component from the paper.
Table 2 enables a lower-tail component: the score-preferred action pays 0.90
unless an X-dependent catastrophe produces zero, while the other action pays
0.79. Covariate shift changes only `P(X)`; concept drift changes only the
catastrophe probability. All six severities attain KL values 0.005, 0.02, 0.045,
0.08, 0.125, and 0.18.

Only the SNLS linear baseline is run. KNLS is excluded.

Run 100 replications from `code/revision`:

```powershell
conda run -n yijun_env python simulation\shift_dgp_search\action_specific_unified\run_experiment.py --num_reps 100
conda run -n yijun_env python simulation\shift_dgp_search\action_specific_unified\summarize.py
conda run -n yijun_env python simulation\shift_dgp_search\action_specific_unified\make_figures.py
conda run -n yijun_env python simulation\shift_dgp_search\action_specific_unified\validate.py
```

The pilot search was frozen before the 100-replication run. `search_log.csv`
records the rejected symmetric action-specific specifications and the reason for
selecting the modular outcome specification.
