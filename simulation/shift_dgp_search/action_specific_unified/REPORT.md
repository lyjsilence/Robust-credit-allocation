# Unified Simulation Results

## Material Passport

- Type: simulation experiment result
- Status: verified
- Replications: 100 independent datasets
- Environment: `yijun_env`, NVIDIA RTX 3090
- Training radius: 0.20
- Baseline: SNLS only; KNLS was not run
- Verification: complete replication IDs 0--99, exact deterministic regeneration

## Table 1. Robust value by policy class

Values are mean ± Monte Carlo standard error.

| Policy class | Model | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear | SNLS | 0.7210 ± 0.0002 | 0.6805 ± 0.0002 | 0.5994 ± 0.0002 | 0.5184 ± 0.0002 | 0.4373 ± 0.0002 |
| Linear | Ours | 0.7201 ± 0.0003 | 0.6795 ± 0.0003 | 0.5985 ± 0.0003 | 0.5174 ± 0.0003 | 0.4364 ± 0.0003 |
| Nonlinear | SNLS | 0.7017 ± 0.0006 | 0.6593 ± 0.0006 | 0.5745 ± 0.0006 | 0.4896 ± 0.0006 | 0.4048 ± 0.0006 |
| Nonlinear | Ours | 0.7213 ± 0.0005 | 0.6807 ± 0.0005 | 0.5995 ± 0.0005 | 0.5184 ± 0.0005 | 0.4372 ± 0.0005 |

The linear policies are nearly indistinguishable. Under the nonlinear S-shaped
boundary, Ours exceeds SNLS at every test radius, and the absolute gap grows
from 0.0196 to 0.0324 as the radius increases.

## Table 2. Robust versus non-robust learning under deployment shift

| Shift | Lambda | KL | Robust | Non-robust | Improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Covariate | 0.5 | 0.005 | 0.7862 ± 0.0002 | 0.7799 ± 0.0008 | 0.81% |
| Covariate | 1.0 | 0.020 | 0.7843 ± 0.0002 | 0.7628 ± 0.0010 | 2.83% |
| Covariate | 1.5 | 0.045 | 0.7826 ± 0.0003 | 0.7450 ± 0.0013 | 5.04% |
| Covariate | 2.0 | 0.080 | 0.7808 ± 0.0003 | 0.7273 ± 0.0016 | 7.35% |
| Covariate | 2.5 | 0.125 | 0.7785 ± 0.0004 | 0.7080 ± 0.0018 | 9.97% |
| Covariate | 3.0 | 0.180 | 0.7763 ± 0.0005 | 0.6874 ± 0.0022 | 12.94% |
| Concept | 0.5 | 0.005 | 0.7897 ± 0.0001 | 0.7912 ± 0.0006 | -0.18% |
| Concept | 1.0 | 0.020 | 0.7876 ± 0.0002 | 0.7757 ± 0.0006 | 1.54% |
| Concept | 1.5 | 0.045 | 0.7853 ± 0.0002 | 0.7592 ± 0.0008 | 3.44% |
| Concept | 2.0 | 0.080 | 0.7828 ± 0.0002 | 0.7413 ± 0.0010 | 5.59% |
| Concept | 2.5 | 0.125 | 0.7800 ± 0.0003 | 0.7224 ± 0.0011 | 7.97% |
| Concept | 3.0 | 0.180 | 0.7771 ± 0.0003 | 0.7029 ± 0.0013 | 10.55% |

Covariate-shift improvement is positive and monotone over all severities.
Under concept drift, the policies are practically close at the smallest KL
(absolute difference 0.0014); robust advantage appears at lambda 1 and then
increases monotonically to 10.55%.

## Table 3. Fixed versus data-driven policy complexity

| Complexity | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=1 | 0.7068 ± 0.0045 | 0.6656 ± 0.0046 | 0.5831 ± 0.0047 | 0.5006 ± 0.0049 | 0.4182 ± 0.0050 |
| K=2 | 0.7167 ± 0.0009 | 0.6758 ± 0.0010 | 0.5941 ± 0.0011 | 0.5124 ± 0.0012 | 0.4307 ± 0.0013 |
| K=3 | 0.7156 ± 0.0031 | 0.6748 ± 0.0031 | 0.5933 ± 0.0033 | 0.5117 ± 0.0034 | 0.4302 ± 0.0035 |
| K=4 | 0.7189 ± 0.0008 | 0.6782 ± 0.0008 | 0.5968 ± 0.0008 | 0.5154 ± 0.0009 | 0.4340 ± 0.0010 |
| K=5 | 0.7189 ± 0.0008 | 0.6782 ± 0.0008 | 0.5967 ± 0.0009 | 0.5153 ± 0.0010 | 0.4339 ± 0.0010 |
| K=6 | 0.7031 ± 0.0067 | 0.6620 ± 0.0068 | 0.5800 ± 0.0070 | 0.4979 ± 0.0073 | 0.4158 ± 0.0075 |
| Data-driven | 0.7213 ± 0.0005 | 0.6807 ± 0.0005 | 0.5995 ± 0.0005 | 0.5184 ± 0.0005 | 0.4372 ± 0.0005 |

The data-driven procedure has the highest mean value at every radius. Selected
depth frequencies were K=1: 42, K=2: 35, K=3: 16, K=4: 5, and K=5: 2.

## Decision boundaries

- `figure_linear.png`, `figure_nonlinear.png`: paper-style observed decision points.
- `figure_linear_boundary.png`, `figure_nonlinear_boundary.png`: the same points
  with the true boundary overlaid.
- `figure_linear_regions.png`, `figure_nonlinear_regions.png`: the same points
  over faint model-specific decision regions.

All panels use the same 2,500 replication-0 test observations and the paper's
purple/yellow action colors. Ours and SNLS both recover the linear boundary;
under the S-shaped boundary, Ours follows the nonlinear structure while SNLS
remains approximately linear.

## Reproducibility checks

- All 100 result files contain exactly 20, 24, and 35 rows for Tables 1, 2,
  and 3, respectively.
- Replication 0 regenerates exactly for the audited arrays.
- Maximum absolute KL error is 3.192e-16 for covariate shift and 1.388e-16
  for concept drift.
- All four 25-replication worker processes exited with code 0.
