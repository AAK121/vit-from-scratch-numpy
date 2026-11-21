"""Plots for the Sep 2026 "strong baseline + ablations" session: recipe A/B, the (unchanged-
format) model/data scaling sweeps re-run under the new v2 recipe, the TinyViT-inspired
structural-change ablation, the DCT tokenizer ablation, the combined-improvements runs, and a
CPU-vs-GPU profiling comparison. Kept separate from `make_plots.py` (which still serves the
original 64px-baseline/old-recipe report figures) rather than editing it in place.

Usage: python scripts/make_plots_v2.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(RESULTS, "plots")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_row(rows, key="val_acc"):
    return max(rows, key=lambda r: float(r[key]))


def plot_recipe_comparison():
    """V1 (minimally-regularized) configuration vs. v2 (literature-informed) configuration,
    same architecture/data/machine. V1 is the control arm of this A/B test -- the held-fixed
    baseline recipe kept unchanged so v2's effect can be attributed to the recipe change alone."""
    v2 = read_csv(os.path.join(RESULTS, "baseline_v2_recipe.csv"))
    v1 = read_csv(os.path.join(RESULTS, "baseline_old_recipe.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for rows, label, color in [(v2, "v2 configuration", "tab:blue"), (v1, "V1 configuration", "tab:red")]:
        ep = [int(r["epoch"]) for r in rows]
        va = [float(r["val_acc"]) for r in rows]
        ta = [float(r["train_acc"]) for r in rows]
        axes[0].plot(ep, va, "-", color=color, label=f"{label} (val)")
        axes[0].plot(ep, ta, "--", color=color, alpha=0.55, label=f"{label} (train)")
        vl = [float(r["val_loss"]) for r in rows]
        axes[1].plot(ep, vl, "-", color=color, label=label)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("accuracy")
    axes[0].set_title("Train/val accuracy"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val loss")
    axes[1].set_title("Validation loss (V1 diverges -- overfitting)"); axes[1].legend(fontsize=8)
    fig.suptitle("Strong-baseline A/B: v2 configuration vs. V1 configuration, same architecture/data/machine")
    fig.tight_layout()
    out = os.path.join(PLOTS, "recipe_comparison_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_model_scaling_v2():
    path = os.path.join(RESULTS, "model_scaling.csv")
    rows = read_csv(path)
    axes_names = ["layers", "heads", "embed_dim"]
    axes_names = [a for a in axes_names if any(r["axis"] == a for r in rows)]

    fig, axs = plt.subplots(1, len(axes_names), figsize=(5 * len(axes_names), 4))
    if len(axes_names) == 1:
        axs = [axs]
    for ax, axis in zip(axs, axes_names):
        sub = [r for r in rows if r["axis"] == axis]
        sub.sort(key=lambda r: float(r["value"]))
        xs = [float(r["value"]) for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        ax.plot(xs, ys, "o-", color="tab:blue")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8,
                        ha="center")
        ax.set_xlabel(axis); ax.set_ylabel("best val accuracy")
        ax.set_title(f"Model scaling: {axis} (v2 recipe, full budget)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_scaling_by_axis_v2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    markers = {"layers": "o", "heads": "s", "embed_dim": "^"}
    for axis in axes_names:
        sub = [r for r in rows if r["axis"] == axis]
        xs = [float(r["FLOPs"]) for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        ax.scatter(xs, ys, label=axis, marker=markers.get(axis, "o"), s=60)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (log scale)"); ax.set_ylabel("best val accuracy")
    ax.set_title("Accuracy vs. compute (FLOPs), v2 recipe, full budget + early stopping")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_scaling_acc_vs_flops_v2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_data_scaling_v2():
    path = os.path.join(RESULTS, "data_scaling.csv")
    rows = read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for sampling, color in [("uniform", "tab:blue"), ("nonuniform", "tab:red")]:
        sub = [r for r in rows if r["sampling"] == sampling]
        sub.sort(key=lambda r: float(r["fraction"]))
        xs = [float(r["fraction"]) * 100 for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        ax.plot(xs, ys, "o-", color=color, label=sampling)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8,
                        ha="center", color=color)
    ax.set_xlabel("training data used (%)"); ax.set_ylabel("best val accuracy")
    ax.set_title("Data scaling, v2 recipe, full budget + early stopping")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "data_scaling_accuracy_v2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


VARIANT_LABELS = {
    "baseline_v2_recipe": "baseline (v2 recipe)",
    "tinyvit_local_only": "+ local-conv",
    "tinyvit_conv_only": "+ conv patch stem",
    "tinyvit_conv_local": "+ conv stem + local-conv",
    "dct_full": "+ DCT tokenizer (full)",
    "dct_trunc16": "+ DCT tokenizer (trunc16)",
    "combo_dct_local": "+ DCT + local-conv",
    "combo_depth6_dct_local": "+ depth6 + DCT + local-conv",
    "scale_layers_6": "+ depth6 alone",
}


def _variant_stats(tag):
    rows = read_csv(os.path.join(RESULTS, f"{tag}.csv"))
    b = best_row(rows)
    return float(b["val_acc"])


def plot_structural_and_dct_ablation():
    order = ["baseline_v2_recipe", "tinyvit_local_only", "tinyvit_conv_only", "tinyvit_conv_local",
             "dct_full", "dct_trunc16"]
    order = [t for t in order if os.path.exists(os.path.join(RESULTS, f"{t}.csv"))]
    vals = [_variant_stats(t) for t in order]
    labels = [VARIANT_LABELS.get(t, t) for t in order]
    baseline_val = vals[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tab:gray"] + ["tab:green" if v > baseline_val else "tab:red" for v in vals[1:]]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(baseline_val, color="black", linestyle="--", linewidth=0.8, label="baseline")
    for i, v in enumerate(vals):
        delta = v - baseline_val
        sign = "+" if delta >= 0 else ""
        ax.annotate(f"{v:.4f}\n({sign}{delta*100:.2f}pt)", (i, v), textcoords="offset points",
                    xytext=(0, 6), fontsize=8, ha="center")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("best val accuracy")
    ax.set_title("Structural-change and DCT-tokenizer ablation vs. v2 baseline (same recipe/data)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "structural_dct_ablation.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_combined_pareto():
    """val_acc vs. FLOPs across every variant tried this session, combo runs included once
    available -- the "which single change/combination is actually worth its compute" view."""
    import json
    tags = ["baseline_v2_recipe", "scale_layers_6", "tinyvit_local_only", "tinyvit_conv_only",
            "tinyvit_conv_local", "dct_full", "dct_trunc16", "combo_dct_local",
            "combo_depth6_dct_local"]
    pts = []
    for t in tags:
        csv_path = os.path.join(RESULTS, f"{t}.csv")
        ckpt_path = os.path.join(ROOT, "checkpoints", t, "best.json")
        if not (os.path.exists(csv_path) and os.path.exists(ckpt_path)):
            print(f"skip (missing): {t}")
            continue
        val_acc = _variant_stats(t)
        with open(ckpt_path) as f:
            meta = json.load(f)
        sys.path.insert(0, ROOT)
        from utils.flops import count_flops_macs
        fm = count_flops_macs(meta["config"])
        pts.append((t, fm["FLOPs"], val_acc))

    fig, ax = plt.subplots(figsize=(8, 6))
    for t, flops, val_acc in pts:
        is_baseline = t == "baseline_v2_recipe"
        is_combo = t.startswith("combo_")
        color = "black" if is_baseline else ("tab:purple" if is_combo else "tab:blue")
        marker = "*" if is_baseline else ("D" if is_combo else "o")
        size = 220 if is_baseline else (160 if is_combo else 90)
        ax.scatter(flops, val_acc, color=color, marker=marker, s=size, zorder=3)
        ax.annotate(VARIANT_LABELS.get(t, t), (flops, val_acc), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("FLOPs (forward pass)")
    ax.set_ylabel("best val accuracy")
    ax.set_title("All changes tried this session: accuracy vs. compute cost")
    fig.tight_layout()
    out = os.path.join(PLOTS, "combined_pareto.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_profiling_comparison():
    """Hardcoded from this session's `scripts/profiling.py` runs (train_logs/profiling_{cpu,gpu}.log)
    -- CPU: AMD EPYC-class conda env; GPU: NVIDIA RTX A6000, both on the current machine."""
    components = ["patch_embed+cls_pos", "block-avg attn(+ln1)", "block-avg mlp(+ln2)", "ln_final+head"]
    cpu_ms = [7.24, 23.52, 69.03, 2.06]
    gpu_ms = [0.216, 0.812, 0.777, 0.346]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(components))
    axes[0].bar(x, cpu_ms, color="tab:orange")
    axes[0].set_xticks(x); axes[0].set_xticklabels(components, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("ms / forward pass"); axes[0].set_title("CPU (this machine's conda env)")

    axes[1].bar(x, gpu_ms, color="tab:green")
    axes[1].set_xticks(x); axes[1].set_xticklabels(components, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("ms / forward pass"); axes[1].set_title("GPU (RTX A6000, CuPy)")
    fig.suptitle("Per-component forward-pass cost, batch 64 (this session's profiling run)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "profiling_comparison_v2.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    plot_recipe_comparison()
    plot_model_scaling_v2()
    plot_data_scaling_v2()
    plot_structural_and_dct_ablation()
    plot_combined_pareto()
    plot_profiling_comparison()


if __name__ == "__main__":
    main()
