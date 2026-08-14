# Dotted True-Boundary Overlay

## Scope

Change only the true-boundary overlay in the two option-2 figures:

- `figure_linear_boundary.png`
- `figure_nonlinear_boundary.png`

## Design

- Draw the true boundary as a black dotted line (`linestyle=":"`).
- Keep the current line width of 1.1.
- Apply the same overlay to the Ground Truth, SNLS, and Ours panels.
- Preserve all observations, decisions, colors, axes, labels, models, data, and
  the other four figure files unchanged.

## Verification

Regenerate the figures, confirm visually that both option-2 boundaries are
dotted, and verify that the four non-boundary PNG files are byte-for-byte
unchanged.
