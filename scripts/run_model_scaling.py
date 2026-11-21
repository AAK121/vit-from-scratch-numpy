"""Core Component 2 -- Model scaling sweep.

Trains the baseline architecture (embed_dim=128, depth=4, num_heads=4) with one axis
(layers / heads / embed dim) varied at a time, dataset fixed at 100%, and logs params,
FLOPs, MACs, train/val performance, and inference latency/throughput for each variant
to results/model_scaling.csv.

Usage: python scripts/run_model_scaling.py --epochs 8
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.transformer import build_vit
from utils.checkpoint import load_into_model
from utils.flops import count_flops_macs
from utils.metrics import measure_latency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_LOCK = threading.Lock()

BASE = dict(embed_dim=128, depth=4, num_heads=4)

SWEEP = []
for depth in [2, 4, 6, 8]:
    cfg = dict(BASE)
    cfg["depth"] = depth
    SWEEP.append(("layers", depth, cfg))
for heads in [2, 4, 8]:
    cfg = dict(BASE)
    cfg["num_heads"] = heads
    SWEEP.append(("heads", heads, cfg))
for dim in [64, 128, 256]:
    cfg = dict(BASE)
    cfg["embed_dim"] = dim
    SWEEP.append(("embed_dim", dim, cfg))


def write_row(path, row):
    with _CSV_LOCK:
        exists = os.path.exists(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)


def run_one(axis, value, overrides, epochs, config_path, out_csv, lr_schedule, early_stop_patience,
            gpu=None):
    tag = f"scale_{axis}_{value}"
    cmd = [sys.executable, os.path.join(ROOT, "train.py"),
           "--config", config_path, "--tag", tag, "--epochs", str(epochs),
           "--lr-schedule", lr_schedule, "--early-stop-patience", str(early_stop_patience)]
    for k, v in overrides.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("=" * 60)
    print(f"[gpu={gpu}] Running:", " ".join(cmd))
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    train_time = time.perf_counter() - t0

    run_csv = os.path.join(ROOT, "results", f"{tag}.csv")
    with open(run_csv) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    best_val_acc = max(float(r["val_acc"]) for r in rows)

    ckpt = os.path.join(ROOT, "checkpoints", tag, "best")
    with open(ckpt + ".json") as f:
        ckpt_meta = json.load(f)
    model_config = ckpt_meta["config"]
    model = build_vit(model_config)
    load_into_model(model, ckpt)
    n_params = model.num_parameters()
    fm = count_flops_macs(model_config)
    lat = measure_latency(
        model,
        (model_config["img_size"], model_config["img_size"], model_config["in_chans"]),
        batch_size=1,
    )

    row = dict(
        axis=axis, value=value, tag=tag,
        params=n_params, MACs=fm["MACs"], FLOPs=fm["FLOPs"],
        train_loss=last["train_loss"], train_acc=last["train_acc"],
        val_loss=last["val_loss"], val_acc=last["val_acc"], best_val_acc=best_val_acc,
        train_time_s=round(train_time, 2),
        latency_ms=round(lat["mean_ms"], 4),
        throughput_img_s=round(lat["throughput_img_s"], 2),
    )
    write_row(out_csv, row)
    print(f"-> {row}")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.json")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--gpus", default="0,1", help="comma-separated CUDA device ids to run on "
                    "in parallel, one subprocess per device; empty string disables pinning")
    p.add_argument("--out", default="results/model_scaling.csv")
    args = p.parse_args()

    out_csv = os.path.join(ROOT, args.out)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""] or [None]

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = []
        for i, (axis, value, overrides) in enumerate(SWEEP):
            gpu = gpus[i % len(gpus)]
            futures.append(ex.submit(
                run_one, axis, value, overrides, args.epochs, args.config, out_csv,
                args.lr_schedule, args.early_stop_patience, gpu))
        for f in futures:
            f.result()

    print(f"\nModel scaling sweep complete -> {out_csv}")


if __name__ == "__main__":
    main()
