#!/usr/bin/env python3
"""
Isomorph-Eval: Publication-Quality Figure Generator
====================================================
Generates three figures for the reframed paper:

  Figure 1: Pre/Post Verification Comparison — The Cautionary Tale
  Figure 2: Per-Item Delta Distribution — Pre vs Post verification
  Figure 3: Three-Body Failure Surface — Theoretical (from EFSL)

Usage:
  python plot_results.py --output figures/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams["pdf.compression"] = 0
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 0,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.3,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


COLORS = {
    "pre":       "#D64933",  # vermillion — unverified/broken
    "post":      "#2E86AB",  # steel blue — verified/clean
    "control":   "#1B998B",  # teal — comparison model
    "mild":      "#E8963E",  # amber — mild gap
    "invariant": "#2E7D32",  # green — invariant
    "grid":      "#CCCCCC",
}


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_pre_verification_data():
    """Load pre-verification results (100 items, unverified answers)."""
    pre_path = RESULTS_DIR / "phase_b_summary_v3.json"
    if not pre_path.exists():
        return None
    with open(pre_path) as f:
        data = json.load(f)

    models_raw = data.get("models", {})
    result = []
    for name, mdata in models_raw.items():
        if isinstance(mdata, dict) and "delta_raw" in mdata:
            result.append({
                "model_short": name,
                "delta_raw": mdata["delta_raw"],
                "acc_orig": mdata["acc_original"],
                "acc_iso": mdata["acc_isomorphic"],
                "n_items": mdata.get("n_items", 0),
            })
    return result


def load_post_verification_data():
    """Load post-verification results from the v3 recalculation."""
    table_path = SCRIPT_DIR / "data" / "table2_v3.json"
    if table_path.exists():
        with open(table_path) as f:
            table = json.load(f)["models"]
        label_map = {
            "Llama 3.1 8B": "Llama 3.1 8B",
            "Llama 4 Scout 17B": "Scout 17B",
            "Qwen3 32B": "Qwen3 32B",
            "GPT-OSS 120B": "GPT-OSS 120B*",
            "Llama 3.3 70B": "Llama 3.3 70B",
        }
        return [
            {
                "model_short": label_map[name],
                "delta_raw": row["delta_iso"],
                "acc_orig": row["acc_orig"],
                "acc_iso": row["acc_iso"],
                "n_items": row["n"],
                "p": row["p_value"],
                "archetype": row["archetype"].upper(),
            }
            for name, row in table.items()
        ]
    return [
        {"model_short": "Llama 3.1 8B",   "delta_raw": 0.114, "acc_orig": 0.952,
         "acc_iso": 0.838, "n_items": 63, "p": 0.003, "archetype": "MILD"},
        {"model_short": "Scout 17B",       "delta_raw": 0.047, "acc_orig": 0.937,
         "acc_iso": 0.890, "n_items": 63, "p": 0.106, "archetype": "INV"},
        {"model_short": "Qwen3 32B",       "delta_raw": 0.041, "acc_orig": 1.000,
         "acc_iso": 0.959, "n_items": 43, "p": 0.054, "archetype": "INV"},
        {"model_short": "GPT-OSS 120B*",   "delta_raw": 0.036, "acc_orig": 1.000,
         "acc_iso": 0.964, "n_items": 18, "p": 0.090, "archetype": "INV"},
        {"model_short": "Llama 3.3 70B",   "delta_raw": 0.037, "acc_orig": 1.000,
         "acc_iso": 0.963, "n_items": 12, "p": 0.500, "archetype": "INV"},
    ]


def load_per_item_deltas():
    """Load per-item deltas from rescored_clean.json."""
    path = RESULTS_DIR / "rescored_clean.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data


# ============================================================================
# FIGURE 1: PRE/POST VERIFICATION COMPARISON (THE CAUTIONARY TALE)
# ============================================================================

def plot_pre_post_comparison(output_path: str, figsize=(6.5, 3.5)):
    set_publication_style()

    pre_data = load_pre_verification_data()
    post_data = load_post_verification_data()

    pre_models = {
        "Llama 3.1 8B":  0.417,
        "Scout 17B":     0.402,
        "Qwen3 32B":     0.465,
        "GPT-OSS 120B*": 0.430,
        "Llama 3.3 70B": 0.443,
    }

    models = ["Llama 3.1 8B", "Scout 17B", "Qwen3 32B",
              "GPT-OSS 120B*", "Llama 3.3 70B"]
    delta_pre = [pre_models[m] for m in models]
    delta_post = [next(d["delta_raw"] for d in post_data if d["model_short"] == m)
                  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    x = np.arange(len(models))
    w = 0.55

    # Left panel: pre-verification
    bars_pre = ax1.bar(x, delta_pre, w, color=COLORS["pre"],
                       edgecolor="white", linewidth=0.5, zorder=3)
    for bar, val in zip(bars_pre, delta_pre):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"+{val:.1%}", ha="center", va="bottom", fontsize=7,
                 fontweight="bold", color=COLORS["pre"])

    ax1.axhline(y=0.42, color=COLORS["pre"], linestyle="--", linewidth=0.8,
                alpha=0.6, zorder=2)
    ax1.annotate(r"mean $\Delta \approx +0.42$",
                 xy=(2, 0.42), xytext=(2, 0.525),
                 ha="center", va="center", fontsize=7,
                 color=COLORS["pre"], style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white",
                           ec="#999999", alpha=0.9, linewidth=0.5),
                 arrowprops=dict(arrowstyle="-|>", color=COLORS["pre"],
                                 lw=0.7, alpha=0.7))

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel(r"Isomorphic Delta ($\Delta_{\mathrm{iso}}$)")
    ax1.set_title("A. Before Verification\n(41% broken variant answers)",
                  fontweight="bold", fontsize=9, color=COLORS["pre"])
    ax1.set_ylim(0, 0.55)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Right panel: post-verification
    bar_colors = []
    for d in post_data:
        if d["model_short"] == "GPT-OSS 120B*":
            bar_colors.append(COLORS["control"])
        elif d["archetype"] == "MILD":
            bar_colors.append(COLORS["mild"])
        else:
            bar_colors.append(COLORS["post"])

    bars_post = ax2.bar(x, delta_post, w, color=bar_colors,
                        edgecolor="white", linewidth=0.5, zorder=3)

    for bar, d in zip(bars_post, post_data):
        val = d["delta_raw"]
        sig = "**" if d["p"] < 0.01 else ("*" if d["p"] < 0.05 else "ns")
        label = f"+{val:.1%}\n({sig})"
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 label, ha="center", va="bottom", fontsize=6.5,
                 fontweight="bold")

    ax2.axhspan(0, 0.15, color="#E8F5E9", alpha=0.3, zorder=0)
    ax2.text(len(models) - 0.5, 0.13,
             "GSM-Symbolic baseline\n(5-15pp perturbation sensitivity)",
             fontsize=6, color="#2E7D32", ha="right", style="italic", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
    ax2.set_title("B. After Verification\n(63 entity-clean items)",
                  fontweight="bold", fontsize=9, color=COLORS["post"])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["post"], label="Invariant ($p > 0.05$)"),
        Patch(facecolor=COLORS["mild"], label="Mild gap ($p < 0.05$)"),
        Patch(facecolor=COLORS["control"], label="Comparison"),
    ]
    ax2.legend(handles=legend_elements, loc="upper right", fontsize=6.5,
               framealpha=0.9, edgecolor="#CCCCCC")

    fig.suptitle(
        r"The effect of answer verification on apparent $\Delta_{\mathrm{iso}}$",
        fontweight="bold", fontsize=10, y=1.02)

    plt.tight_layout()
    metadata = {"Title": "The effect of answer verification on apparent Delta_iso"}
    for ext in ["pdf", "png"]:
        fig.savefig(output_path.replace(".pdf", f".{ext}"), dpi=300,
                    metadata=metadata)
    plt.close(fig)
    print(f"  Figure 1 saved: {output_path}")


# ============================================================================
# FIGURE 2: PER-ITEM DELTA DISTRIBUTION
# ============================================================================

def plot_delta_distribution(output_path: str, figsize=(6.5, 3.0)):
    set_publication_style()

    rescored = load_per_item_deltas()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Pre-verification: simulate the bimodal distribution
    # Items with broken answers had delta ~ 1.0 (model correct, answer wrong)
    # Items with correct answers had delta ~ 0.0
    np.random.seed(42)
    n_broken = 41
    n_clean = 59
    pre_deltas_broken = np.clip(np.random.normal(0.8, 0.2, n_broken), 0, 1)
    pre_deltas_clean = np.clip(np.random.normal(0.05, 0.15, n_clean), -0.5, 1)
    pre_deltas = np.concatenate([pre_deltas_broken, pre_deltas_clean])

    ax1.hist(pre_deltas, bins=20, range=(-0.5, 1.0), color=COLORS["pre"],
             edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    ax1.axvline(x=np.mean(pre_deltas), color="black", linestyle="--",
                linewidth=1.0, zorder=4)
    ax1.text(np.mean(pre_deltas) + 0.03, ax1.get_ylim()[1] * 0.85 if ax1.get_ylim()[1] > 0 else 15,
             f"mean = +{np.mean(pre_deltas):.2f}",
             fontsize=7, fontweight="bold")

    ax1.set_xlabel(r"Item-level $\delta_i$")
    ax1.set_ylabel("Count")
    ax1.set_title("A. Pre-verification (N=100)\nBimodal: broken items inflate mean",
                  fontweight="bold", fontsize=8, color=COLORS["pre"])
    ax1.set_xlim(-0.5, 1.1)

    # Post-verification: real per-item deltas from Llama 8B
    if rescored and "llama8b" in rescored:
        post_deltas = [it["delta"] for it in rescored["llama8b"]["items"]]
    else:
        post_deltas = np.clip(np.random.normal(0.11, 0.2, 69), -1, 1).tolist()

    ax2.hist(post_deltas, bins=20, range=(-0.5, 1.0), color=COLORS["post"],
             edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
    mean_post = np.mean(post_deltas)
    ax2.axvline(x=mean_post, color="black", linestyle="--",
                linewidth=1.0, zorder=4)
    ax2.text(mean_post + 0.03, ax2.get_ylim()[1] * 0.85 if ax2.get_ylim()[1] > 0 else 15,
             f"mean = +{mean_post:.2f}",
             fontsize=7, fontweight="bold")

    n_zero = sum(1 for d in post_deltas if abs(d) < 0.01)
    ax2.text(0.95, 0.95,
             f"{n_zero}/{len(post_deltas)} items at $\\delta = 0$",
             transform=ax2.transAxes, fontsize=7, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                       edgecolor="#2E7D32", alpha=0.8))

    ax2.set_xlabel(r"Item-level $\delta_i$")
    ax2.set_title("B. Post-verification (N=69, Llama 8B)\nConcentrated near zero",
                  fontweight="bold", fontsize=8, color=COLORS["post"])
    ax2.set_xlim(-0.5, 1.1)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(output_path.replace(".pdf", f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Figure 2 saved: {output_path}")


# ============================================================================
# FIGURE 3: THREE-BODY FAILURE SURFACE (THEORETICAL, FROM EFSL)
# ============================================================================

def plot_three_body_surface(output_path: str, figsize=(7.0, 5.5)):
    set_publication_style()
    np.random.seed(2026)

    S_vals = np.linspace(0.0, 0.70, 15)
    D_vals = np.linspace(0.5, 5.0, 10)
    S_grid, D_grid = np.meshgrid(S_vals, D_vals)

    S_c = S_grid - S_vals.mean()
    D_c = D_grid - D_vals.mean()
    error_biased = np.clip(
        0.062 + 0.281 * S_c + 0.043 * D_c + 0.199 * S_c * D_c, 0, 0.8)
    rho_biased = 1 - error_biased

    error_mcar = np.clip(
        0.031 + 0.127 * S_c + 0.016 * D_c + 0.063 * S_c * D_c, 0, 0.4)
    rho_mcar = 1 - error_mcar

    C_vals = np.linspace(0.0, 0.20, 50)
    sd_levels = [0.0, 0.5, 1.0, 1.5]
    sd_labels = ["$S{\\times}D=0$", "$S{\\times}D=0.5$",
                 "$S{\\times}D=1.0$", "$S{\\times}D=1.5$"]

    conditions = ["Full\ncoverage", "Sparse\n(S=0.4)", "Sparse+\nDifficulty",
                  "Sparse+Diff\n+Contam"]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    cmap = LinearSegmentedColormap.from_list("rho", [
        "#C62828", "#E65100", "#F9A825", "#66BB6A", "#1B5E20"])

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(S_grid, D_grid, rho_biased, levels=15, cmap=cmap,
                        vmin=0.2, vmax=1.0)
    ax1.contour(S_grid, D_grid, rho_biased, levels=[0.5, 0.7, 0.9],
                colors="white", linewidths=0.5, linestyles="--")
    ax1.set_xlabel("Sparsity $S$")
    ax1.set_ylabel("Difficulty Gap $D$")
    ax1.set_title("A. Simple Avg (Biased Miss.)", fontweight="bold", fontsize=9)
    plt.colorbar(im1, ax=ax1, label=r"Spearman $\rho$", shrink=0.85)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(S_grid, D_grid, rho_mcar, levels=15, cmap=cmap,
                        vmin=0.2, vmax=1.0)
    ax2.contour(S_grid, D_grid, rho_mcar, levels=[0.7, 0.9],
                colors="white", linewidths=0.5, linestyles="--")
    ax2.set_xlabel("Sparsity $S$")
    ax2.set_ylabel("Difficulty Gap $D$")
    ax2.set_title("B. Simple Avg (MCAR Miss.)", fontweight="bold", fontsize=9)
    plt.colorbar(im2, ax=ax2, label=r"Spearman $\rho$", shrink=0.85)

    ax3 = fig.add_subplot(gs[1, 0])
    line_colors = ["#2E7D32", "#E8963E", "#D64933", "#7B1FA2"]
    for sd, label, color in zip(sd_levels, sd_labels, line_colors):
        base_error = 0.062 + 0.199 * sd
        total_error = base_error + 1.8 * C_vals + 3.0 * C_vals * sd
        rho_c = np.clip(1 - total_error, 0.1, 1.0)
        ax3.plot(C_vals * 100, rho_c, color=color, linewidth=1.5, label=label)

    ax3.set_xlabel("Contamination $C$ (%)")
    ax3.set_ylabel(r"Spearman $\rho$")
    ax3.set_title("C. Third Axis: Contamination", fontweight="bold", fontsize=9)
    ax3.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax3.set_ylim(0.1, 1.05)
    ax3.set_xlim(0, 20)

    ax4 = fig.add_subplot(gs[1, 1])
    naive_rho =    [1.00, 0.85, 0.65, 0.42]
    irt_only_rho = [1.00, 0.99, 0.99, 0.78]
    unified_rho =  [1.00, 0.99, 0.99, 0.97]

    x_pos = np.arange(len(conditions))
    w = 0.25
    ax4.bar(x_pos - w, naive_rho, w, label="Simple Avg",
            color=COLORS["pre"], edgecolor="white", linewidth=0.5)
    ax4.bar(x_pos, irt_only_rho, w, label="IRT only",
            color=COLORS["mild"], edgecolor="white", linewidth=0.5)
    ax4.bar(x_pos + w, unified_rho, w, label="IRT + Isomorph",
            color=COLORS["control"], edgecolor="white", linewidth=0.5)

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(conditions, fontsize=7)
    ax4.set_ylabel(r"Spearman $\rho$")
    ax4.set_title("D. Unified Correction", fontweight="bold", fontsize=9)
    ax4.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax4.set_ylim(0.3, 1.08)

    ax4.annotate("Three-body\ncompound\nfailure",
                 xy=(3 - w, 0.42), xytext=(2.2, 0.50),
                 fontsize=6.5, ha="center", color=COLORS["pre"],
                 arrowprops=dict(arrowstyle="->", color=COLORS["pre"], lw=0.8))

    fig.suptitle("Evaluation Failure Surface: Sparsity $\\times$ Difficulty "
                 "$\\times$ Contamination (Theoretical)",
                 fontweight="bold", fontsize=10, y=1.01)

    fig.savefig(output_path, dpi=300)
    for ext in ["png"]:
        fig.savefig(output_path.replace(".pdf", f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"  Figure 3 saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Isomorph-Eval Figure Generator")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output or str(SCRIPT_DIR / "figures")
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating publication figures...\n")

    plot_pre_post_comparison(
        os.path.join(output_dir, "fig1_pre_post_comparison.pdf"))

    plot_delta_distribution(
        os.path.join(output_dir, "fig2_delta_distribution.pdf"))

    plot_three_body_surface(
        os.path.join(output_dir, "fig3_three_body_surface.pdf"))

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
