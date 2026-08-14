# Decision-Boundary Figure Variants

## Objective

Replace the grid-filled decision-boundary figures with journal-style figures
that use the actual 2,500 test observations from replication 0. Produce three
styles for both the linear and nonlinear DGP, for six figures in total.

## Shared visual specification

- Use the same test `X` in every panel and every style.
- Ground truth is the score-implied optimal action; SNLS and Ours are their
  learned assignments for the same observations.
- Panel order is `(a) Ground Truth`, `(b) SNLS`, `(c) Ours`.
- Match the paper's discrete viridis endpoints: action 0 is `#440154` (purple)
  and action 1 is `#FDE725` (yellow).
- Use small semi-transparent points, square axes, limits `[-1.05, 1.05]`, and
  mathematical labels for `X1` and `X2`.

## Variants

1. Paper style: points only. Save as `figure_linear.png` and
   `figure_nonlinear.png`.
2. Boundary style: the same points plus the true zero contour. Save with the
   suffix `_boundary.png`.
3. Region style: the same points over faint panel-specific decision regions.
   Save with the suffix `_regions.png`.

## Verification

Confirm that every figure contains exactly three panels based on the same 2,500
observations, visually inspect all six PNGs, and ensure the nonlinear Ours panel
recovers the S shape while SNLS remains approximately linear.
