"""Plot true and learned linear and S-shaped decision boundaries."""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SIMULATION_DIR = HERE.parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

import optimization_problem_with_holdout as optimization
from dgp import decision_score


def model_path(root, boundary, model, depth):
    folder = f"{boundary}_ours_robust" if model == "Ours" else f"{boundary}_snls"
    patterns = (
        f"*exp_id_0_{depth}.pth" if model == "Ours"
        else f"*SNLN-exp_id_0_{depth}.pth"
    )
    return next((root / "models" / "rep-0" / folder).glob(patterns))


def load_model(root, boundary, model, depth, device):
    if model == "Ours":
        policy = optimization.NN_Sieves({}, 5, device, depth, 2)
    else:
        policy = optimization.NN_baseline({}, 5, device, depth, 2)
    policy.load_state_dict(torch.load(model_path(root, boundary, model, depth), map_location=device))
    policy.eval()
    return policy


def plot_boundary(root, boundary, selected_depth, device):
    axis = np.linspace(-1.0, 1.0, 301)
    x1, x2 = np.meshgrid(axis, axis)
    inside = x1**2 + x2**2 <= 1.0
    x = np.column_stack([x1.ravel(), x2.ravel(), np.zeros((x1.size, 3))])
    truth = (decision_score(x, boundary) >= 0).reshape(x1.shape)

    panels = [("Ground truth", truth)]
    for name, depth in (("Ours", selected_depth), ("SNLS", 2)):
        policy = load_model(root, boundary, name, depth, device)
        with torch.no_grad():
            decision = policy(torch.tensor(x, dtype=torch.float32, device=device)).argmax(1)
        panels.append((name, decision.cpu().numpy().reshape(x1.shape)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, (title, decision) in zip(axes, panels):
        shown = np.where(inside, decision, np.nan)
        ax.pcolormesh(x1, x2, shown, cmap="coolwarm", shading="auto", vmin=0, vmax=1)
        ax.contour(x1, x2, truth, levels=[0.5], colors="black", linewidths=1.5)
        ax.set(title=title, xlabel=r"$X_1$", ylabel=r"$X_2$", aspect="equal")
    path = root / f"figure_{boundary}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox_dir", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.sandbox_dir.resolve()
    result = joblib.load(root / "results" / "rep-0.pkl")
    depths = {
        "linear": result["linear_selected_depth"],
        "nonlinear": result["selected_depth"],
    }
    for boundary in ("linear", "nonlinear"):
        plot_boundary(root, boundary, depths[boundary], torch.device(args.device))


if __name__ == "__main__":
    main()
