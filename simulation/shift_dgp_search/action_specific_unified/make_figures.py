"""Plot true and learned decisions using the generated test observations."""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SIMULATION_DIR = HERE.parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

import optimization_problem_with_holdout as optimization
from dgp import decision_score


ACTION_COLORS = ("#440154", "#FDE725")
ACTION_CMAP = ListedColormap(ACTION_COLORS)
PANEL_NAMES = ("Ground Truth", "SNLS", "Ours")


def model_path(root, boundary, model, depth):
    folder = f"{boundary}_ours_robust" if model == "Ours" else f"{boundary}_snls"
    pattern = f"*exp_id_0_{depth}.pth" if model == "Ours" else f"*SNLN-exp_id_0_{depth}.pth"
    return next((root / "models" / "rep-0" / folder).glob(pattern))


def load_model(root, boundary, model, depth, device):
    policy_class = optimization.NN_Sieves if model == "Ours" else optimization.NN_baseline
    policy = policy_class({}, 5, device, depth, 2)
    policy.load_state_dict(
        torch.load(model_path(root, boundary, model, depth), map_location=device)
    )
    policy.eval()
    return policy


def predict(policy, x, device):
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
        return policy(tensor).argmax(1).cpu().numpy()


def make_grid(boundary, policies, device):
    axis = np.linspace(-1.0, 1.0, 301)
    x1, x2 = np.meshgrid(axis, axis)
    x = np.column_stack((x1.ravel(), x2.ravel(), np.zeros((x1.size, 3))))
    score = decision_score(x, boundary).reshape(x1.shape)
    decisions = [(score >= 0).astype(int)]
    decisions.extend(predict(policy, x, device).reshape(x1.shape) for policy in policies)
    return x1, x2, score, decisions


def scatter_decisions(ax, x, decisions):
    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=decisions,
        cmap=ACTION_CMAP,
        vmin=0,
        vmax=1,
        s=6,
        alpha=0.85,
        edgecolors="none",
        rasterized=True,
    )


def format_panel(ax, index, name):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.text(
        0.5,
        -0.20,
        f"({chr(97 + index)}) {name}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#444444", width=0.7, labelsize=9)


def draw_figure(root, boundary, x, point_decisions, grid, variant):
    x1, x2, true_score, grid_decisions = grid
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.45))

    for index, (ax, name, decisions) in enumerate(zip(axes, PANEL_NAMES, point_decisions)):
        if variant == "regions":
            inside = x1**2 + x2**2 <= 1.0
            region = np.ma.masked_where(~inside, grid_decisions[index])
            ax.contourf(
                x1,
                x2,
                region,
                levels=(-0.5, 0.5, 1.5),
                cmap=ACTION_CMAP,
                alpha=0.14,
                antialiased=True,
            )
        scatter_decisions(ax, x, decisions)
        if variant == "boundary":
            ax.contour(x1, x2, true_score, levels=(0,), colors="black", linewidths=1.1)
        format_panel(ax, index, name)

    fig.subplots_adjust(left=0.065, right=0.99, top=0.97, bottom=0.23, wspace=0.27)
    suffix = "" if variant == "paper" else f"_{variant}"
    path = root / f"figure_{boundary}{suffix}.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def plot_boundary(root, boundary, selected_depth, device, data):
    x = data[boundary]["test"]["X"]
    policies = (
        load_model(root, boundary, "SNLS", 2, device),
        load_model(root, boundary, "Ours", selected_depth, device),
    )
    point_decisions = [(decision_score(x, boundary) >= 0).astype(int)]
    point_decisions.extend(predict(policy, x, device) for policy in policies)
    grid = make_grid(boundary, policies, device)

    for variant in ("paper", "boundary", "regions"):
        draw_figure(root, boundary, x, point_decisions, grid, variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sandbox_dir", type=Path, default=HERE)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = args.sandbox_dir.resolve()
    data = joblib.load(root / "data" / "rep-0.pkl")
    result = joblib.load(root / "results" / "rep-0.pkl")
    depths = {
        "linear": result["linear_selected_depth"],
        "nonlinear": result["selected_depth"],
    }
    for boundary in ("linear", "nonlinear"):
        plot_boundary(root, boundary, depths[boundary], torch.device(args.device), data)


if __name__ == "__main__":
    main()
