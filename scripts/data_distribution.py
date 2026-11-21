"""Characterizes the raw `train/` dataset: per-class image counts (are the 100 classes
balanced?) and native image dimensions before the pipeline's resize+center-crop (how much
work is `load_and_resize` actually doing?). Produces `results/plots/data_distribution.png`.

Usage:
    python scripts/data_distribution.py
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from data.dataset import IMG_EXTS, discover_classes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="train")
    p.add_argument("--sample-per-class", type=int, default=5,
                    help="how many images per class to open for the native-size distribution "
                         "(reading all 50k headers is unnecessary for a distribution estimate)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/plots/data_distribution.png")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    classes = discover_classes(args.data_root)

    counts = {}
    for c in classes:
        cdir = os.path.join(args.data_root, c)
        counts[c] = len([f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTS)])
    count_vals = np.array(list(counts.values()))

    sizes = []
    for c in classes:
        cdir = os.path.join(args.data_root, c)
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTS))
        sample = random.sample(files, min(args.sample_per_class, len(files)))
        for f in sample:
            with Image.open(os.path.join(cdir, f)) as im:
                sizes.append(im.size)
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])

    print(f"classes: {len(classes)}  total images: {count_vals.sum()}")
    print(f"per-class count: min={count_vals.min()} max={count_vals.max()} "
          f"mean={count_vals.mean():.1f} std={count_vals.std():.2f}")
    print(f"sampled {len(sizes)} native image sizes ({args.sample_per_class}/class)")
    print(f"native width : min={w.min()} max={w.max()} mean={w.mean():.1f} median={np.median(w):.0f}")
    print(f"native height: min={h.min()} max={h.max()} mean={h.mean():.1f} median={np.median(h):.0f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    order = np.argsort(list(counts.keys()))
    class_names = np.array(list(counts.keys()))[order]
    vals = count_vals[order]
    axes[0].bar(range(len(vals)), vals, width=1.0, color="tab:blue")
    axes[0].set_xlabel("class (sorted by synset ID)")
    axes[0].set_ylabel("images in train/")
    axes[0].set_title(f"Per-class image count\n(min={count_vals.min()}, max={count_vals.max()}, "
                       f"std={count_vals.std():.1f})")
    axes[0].set_ylim(0, count_vals.max() * 1.15)

    axes[1].scatter(w, h, s=8, alpha=0.4, color="tab:orange")
    axes[1].set_xlabel("native width (px)")
    axes[1].set_ylabel("native height (px)")
    axes[1].set_title(f"Native image size before resize\n({len(sizes)} sampled, "
                       f"{len(set(sizes))} unique sizes)")
    axes[1].axvline(64, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(64, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
