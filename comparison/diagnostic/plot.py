### done by claude
### i dont like matplotlib pyplot 

"""
diagnostic/plot.py

Visualizes the baseline divergence map from run_diagnostic.py.
Produces a per-layer cosine similarity bar chart and saves it as PNG.

Usage:
    python diagnostic/plot.py --results logs/diagnostic/diagnostic_results.json
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def plot_similarity(results: dict, output_dir: str):
    similarities = {int(k): v for k, v in results["mean_similarities"].items()}
    layer_map = {int(k): v for k, v in results["layer_map"].items()}
    n_prompts = results["n_prompts"]

    student_layers = sorted(similarities.keys())
    sim_values = [similarities[i] for i in student_layers]
    teacher_layers = [layer_map[i] for i in student_layers]

    divergence = [1 - s for s in sim_values]

    # color: green = similar, red = divergent
    cmap = plt.cm.RdYlGn
    colors = [cmap(s) for s in sim_values]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Baseline Divergence Map — Quasar-3B (student) vs Qwen3-4B (teacher)\n"
        f"n={n_prompts} prompts | projection={'trained' if results['projection_trained'] else 'untrained (random init)'}",
        fontsize=12,
        fontweight="bold",
    )

    # --- plot 1: cosine similarity per layer ---
    ax1 = axes[0]
    bars = ax1.bar(student_layers, sim_values, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("student layer")
    ax1.set_ylabel("cosine similarity")
    ax1.set_title("cosine similarity (student projected → teacher space)")
    ax1.set_xticks(student_layers)
    ax1.set_xticklabels(
        [f"s{i}\n→t{teacher_layers[i]}" for i in student_layers],
        fontsize=7,
    )
    ax1.set_ylim(0, 1)
    ax1.axhline(y=np.mean(sim_values), color="blue", linestyle="--", alpha=0.5,
                label=f"mean = {np.mean(sim_values):.3f}")
    ax1.legend()

    # annotate min (most divergent)
    min_idx = int(np.argmin(sim_values))
    ax1.annotate(
        f"most divergent\ns{student_layers[min_idx]}→t{teacher_layers[student_layers[min_idx]]}",
        xy=(student_layers[min_idx], sim_values[min_idx]),
        xytext=(student_layers[min_idx] + 1.5, sim_values[min_idx] + 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red",
        fontsize=8,
    )

    # --- plot 2: divergence (1 - similarity) ---
    ax2 = axes[1]
    div_colors = [cmap(1 - d) for d in divergence]
    ax2.bar(student_layers, divergence, color=div_colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("student layer")
    ax2.set_ylabel("divergence (1 - cosine sim)")
    ax2.set_title("divergence per layer — training will target layers with highest bars")
    ax2.set_xticks(student_layers)
    ax2.set_xticklabels(
        [f"s{i}" for i in student_layers],
        fontsize=8,
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "divergence_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"plot saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default="logs/diagnostic/diagnostic_results.json")
    parser.add_argument("--output_dir", type=str, default="logs/diagnostic")
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    plot_similarity(results, args.output_dir)
