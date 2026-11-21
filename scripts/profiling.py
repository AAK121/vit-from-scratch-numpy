"""Profiles the from-scratch ViT's forward pass to find the dominant runtime cost (bonus:
Transformer Profiling and Optimization), two ways:

  1. Per-component wall-clock breakdown (patch embedding, each block's attention sublayer,
     each block's MLP sublayer, final norm + head), averaged over many batches, on whichever
     backend `nn/backend.py` selects (forced via VIT_DEVICE) -- with `synchronize()` around
     every timed region so GPU kernel-launch queueing doesn't hide behind the timer.
  2. Python-level cProfile of the same forward pass, to see which individual functions/lines
     actually consume the wall-clock time (confirms the breakdown isn't hiding overhead in,
     e.g., Python-loop bookkeeping rather than the matmuls themselves).

Usage:
    VIT_DEVICE=cpu python scripts/profile.py --checkpoint checkpoints/submission
    VIT_DEVICE=gpu python scripts/profile.py --checkpoint checkpoints/submission
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.backend import DEVICE, synchronize, to_device
from nn.transformer import build_vit
from utils.checkpoint import load_into_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/submission")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-iters", type=int, default=15)
    return p.parse_args()


def timed_forward(model, x):
    """Replicates VisionTransformer.forward, but with a synchronize()d timer bracketing each
    named component, returning a dict of component -> elapsed seconds for this one pass."""
    t = {}

    t0 = time.perf_counter()
    h = model.patch_embed.forward(x, training=False)
    h = model.cls_pos.forward(h, training=False)
    synchronize()
    t["patch_embed+cls_pos"] = time.perf_counter() - t0

    for i, blk in enumerate(model.blocks):
        t0 = time.perf_counter()
        hn = blk.ln1.forward(h, training=False)
        hn = blk.attn.forward(hn, training=False)
        h = h + hn
        synchronize()
        t[f"block{i}.attn(+ln1)"] = t.get(f"block{i}.attn(+ln1)", 0.0) + (time.perf_counter() - t0)

        t0 = time.perf_counter()
        hn = blk.ln2.forward(h, training=False)
        hn = blk.fc1.forward(hn, training=False)
        hn = blk.act.forward(hn, training=False)
        hn = blk.fc2.forward(hn, training=False)
        h = h + hn
        synchronize()
        t[f"block{i}.mlp(+ln2)"] = t.get(f"block{i}.mlp(+ln2)", 0.0) + (time.perf_counter() - t0)

    t0 = time.perf_counter()
    h = model.ln_final.forward(h, training=False)
    logits = model.head.forward(h[:, 0, :], training=False)
    synchronize()
    t["ln_final+head"] = time.perf_counter() - t0

    return t, logits


def main():
    args = parse_args()
    with open(args.checkpoint + ".json") as f:
        ckpt_meta = json.load(f)
    config = ckpt_meta["config"]

    model = build_vit(config)
    load_into_model(model, args.checkpoint)

    rng = np.random.default_rng(0)
    x = to_device(rng.standard_normal(
        (args.batch_size, config["img_size"], config["img_size"], config["in_chans"])
    ).astype(np.float32))

    print(f"backend: {DEVICE}  batch_size: {args.batch_size}  iters: {args.n_iters}")

    for _ in range(args.n_warmup):
        timed_forward(model, x)

    totals = {}
    for _ in range(args.n_iters):
        t, _ = timed_forward(model, x)
        for k, v in t.items():
            totals[k] = totals.get(k, 0.0) + v

    grand_total = sum(totals.values())
    print("\n--- Per-component wall-clock breakdown (mean over "
          f"{args.n_iters} forward passes) ---")
    print(f"{'component':<22}{'ms/pass':>12}{'% of total':>12}")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        ms = 1000.0 * v / args.n_iters
        pct = 100.0 * v / grand_total
        print(f"{k:<22}{ms:>12.4f}{pct:>12.1f}")
    print(f"{'TOTAL':<22}{1000.0*grand_total/args.n_iters:>12.4f}{100.0:>12.1f}")

    print("\n--- cProfile, sorted by internal (tottime) time: which primitive ops actually "
          "burn CPU time (10 forward passes) ---")
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(10):
        timed_forward(model, x)
    profiler.disable()
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    stats.print_stats(12)
    lines = buf.getvalue().splitlines()
    # drop pstats' leading blank line + the "Ordered by:" line; keep the header + top rows
    lines = [ln for ln in lines if ln.strip()]
    header_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith("ncalls"))
    print("\n".join(lines[:1] + lines[header_idx:header_idx + 13]))


if __name__ == "__main__":
    main()
