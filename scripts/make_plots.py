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


def plot_training_curve(tag):
    path = os.path.join(RESULTS, f"{tag}.csv")
    if not os.path.exists(path):
        print(f"skip (missing): {path}")
        return
    rows = read_csv(path)
    epochs = [int(r["epoch"]) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) for r in rows]
    va_acc = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title(f"{tag}: loss")
    axes[0].legend()
    axes[1].plot(epochs, tr_acc, label="train")
    axes[1].plot(epochs, va_acc, label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].set_title(f"{tag}: accuracy")
    axes[1].legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, f"{tag}_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_model_scaling():
    path = os.path.join(RESULTS, "model_scaling.csv")
    if not os.path.exists(path):
        print(f"skip (missing): {path}")
        return
    rows = read_csv(path)
    axes_names = sorted(set(r["axis"] for r in rows))

    fig, axs = plt.subplots(1, len(axes_names), figsize=(5 * len(axes_names), 4))
    if len(axes_names) == 1:
        axs = [axs]
    for ax, axis in zip(axs, axes_names):
        sub = [r for r in rows if r["axis"] == axis]
        sub.sort(key=lambda r: float(r["value"]))
        xs = [float(r["value"]) for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        ax.plot(xs, ys, "o-")
        ax.set_xlabel(axis); ax.set_ylabel("best val accuracy"); ax.set_title(f"Model scaling: {axis}")
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_scaling_by_axis.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    fig, ax = plt.subplots(figsize=(6, 5))
    markers = {"layers": "o", "heads": "s", "embed_dim": "^"}
    for axis in axes_names:
        sub = [r for r in rows if r["axis"] == axis]
        xs = [float(r["FLOPs"]) for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        ax.scatter(xs, ys, label=axis, marker=markers.get(axis, "o"))
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (log scale)"); ax.set_ylabel("best val accuracy")
    ax.set_title("Accuracy vs compute (FLOPs)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_scaling_acc_vs_flops.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for axis in axes_names:
        sub = [r for r in rows if r["axis"] == axis]
        sub.sort(key=lambda r: float(r["value"]))
        xs = [float(r["value"]) for r in sub]
        ys = [float(r["throughput_img_s"]) for r in sub]
        ax.plot(xs, ys, "o-", label=axis)
    ax.set_xlabel("axis value"); ax.set_ylabel("throughput (img/s, batch=1)")
    ax.set_title("Inference throughput vs model scale")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "model_scaling_throughput.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_data_scaling():
    path = os.path.join(RESULTS, "data_scaling.csv")
    if not os.path.exists(path):
        print(f"skip (missing): {path}")
        return
    rows = read_csv(path)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for sampling in ["uniform", "nonuniform"]:
        sub = [r for r in rows if r["sampling"] == sampling]
        sub.sort(key=lambda r: float(r["fraction"]))
        xs = [float(r["fraction"]) for r in sub]
        ys = [float(r["best_val_acc"]) for r in sub]
        axs[0].plot(xs, ys, "o-", label=sampling)
        axs[1].loglog(xs, ys, "o-", label=sampling)
    axs[0].set_xlabel("data fraction"); axs[0].set_ylabel("best val accuracy")
    axs[0].set_title("Data scaling"); axs[0].legend()
    axs[1].set_xlabel("data fraction (log)"); axs[1].set_ylabel("best val accuracy (log)")
    axs[1].set_title("Data scaling (log-log)"); axs[1].legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "data_scaling_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for sampling in ["uniform", "nonuniform"]:
        sub = [r for r in rows if r["sampling"] == sampling]
        sub.sort(key=lambda r: float(r["fraction"]))
        xs = [float(r["fraction"]) for r in sub]
        ys = [float(r["train_time_s"]) for r in sub]
        ax.plot(xs, ys, "o-", label=sampling)
    ax.set_xlabel("data fraction"); ax.set_ylabel("training time (s)")
    ax.set_title("Training time vs data scale"); ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS, "data_scaling_time.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for tag in ["baseline"]:
        plot_training_curve(tag)
    plot_model_scaling()
    plot_data_scaling()


if __name__ == "__main__":
    main()
