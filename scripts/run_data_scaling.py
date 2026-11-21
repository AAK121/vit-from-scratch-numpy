"""Core Component 3 -- Data scaling sweep.

Trains the fixed baseline architecture on 10/25/50/100% of the training pool, under both
uniform (stratified) and non-uniform (Dirichlet-skewed) sampling, and logs training time +
train/val performance to results/data_scaling.csv.

Usage: python scripts/run_data_scaling.py --epochs 8
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_LOCK = threading.Lock()

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SAMPLINGS = ["uniform", "nonuniform"]


def write_row(path, row):
    with _CSV_LOCK:
        exists = os.path.exists(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)


def run_one(fraction, sampling, epochs, config_path, out_csv, lr_schedule, early_stop_patience,
            sample_seed=0, gpu=None):
    tag = f"data_{sampling}_{int(round(fraction * 100))}"
    cmd = [sys.executable, os.path.join(ROOT, "train.py"),
           "--config", config_path, "--tag", tag, "--epochs", str(epochs),
           "--data-fraction", str(fraction), "--sampling", sampling,
           "--sample-seed", str(sample_seed),
           "--lr-schedule", lr_schedule, "--early-stop-patience", str(early_stop_patience)]
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
    train_n = ckpt_meta.get("extra", {}).get("train_n")

    row = dict(
        fraction=fraction, sampling=sampling, tag=tag, train_n=train_n,
        train_loss=last["train_loss"], train_acc=last["train_acc"],
        val_loss=last["val_loss"], val_acc=last["val_acc"], best_val_acc=best_val_acc,
        train_time_s=round(train_time, 2),
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
    p.add_argument("--out", default="results/data_scaling.csv")
    p.add_argument("--sample-seed", type=int, default=0)
    args = p.parse_args()

    out_csv = os.path.join(ROOT, args.out)
    if os.path.exists(out_csv):
        os.remove(out_csv)

    jobs = []
    for fraction in FRACTIONS:
        for sampling in SAMPLINGS:
            if fraction >= 0.999 and sampling == "nonuniform":
                continue  # 100% pool is identical under both sampling strategies
            jobs.append((fraction, sampling))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""] or [None]

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = []
        for i, (fraction, sampling) in enumerate(jobs):
            gpu = gpus[i % len(gpus)]
            futures.append(ex.submit(
                run_one, fraction, sampling, args.epochs, args.config, out_csv,
                args.lr_schedule, args.early_stop_patience, args.sample_seed, gpu))
        for f in futures:
            f.result()

    print(f"\nData scaling sweep complete -> {out_csv}")


if __name__ == "__main__":
    main()
