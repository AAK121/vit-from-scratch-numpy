"""
    bash setup.bash
    python test.py --test-path <absolute-path-to-dataset>

"""
import argparse
import json
import os
import time

from data.dataset import TestLoader
from nn.transformer import build_vit
from utils.checkpoint import load_into_model
from utils.flops import count_flops_macs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "submission")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-path", type=str, required=True,
                    help="absolute path to the test dataset (ImageFolder layout)")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                    help="checkpoint base path (expects <path>.npz and <path>.json)")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.checkpoint + ".json") as f:
        ckpt_meta = json.load(f)
    config = ckpt_meta["config"]
    extra = ckpt_meta.get("extra", {})
    classes = extra["classes"]
    mean = extra["mean"]
    std = extra["std"]

    model = build_vit(config)
    load_into_model(model, args.checkpoint)

    n_params = model.num_parameters()
    fm = count_flops_macs(config)

    loader = TestLoader(args.test_path, classes, config["img_size"], mean, std,
                         batch_size=args.batch_size)
    n_total = len(loader)
    if n_total == 0:
        raise RuntimeError(f"No images found under test-path: {args.test_path}")

    t0 = time.perf_counter()
    n_correct = 0
    for imgs, labels in loader:
        logits = model.forward(imgs, training=False)
        preds = logits.argmax(axis=-1)
        n_correct += int((preds == labels).sum())
    elapsed = time.perf_counter() - t0

    accuracy = n_correct / n_total
    throughput = n_total / elapsed if elapsed > 0 else float("inf")

    print("=" * 55)
    print(f"Test-set accuracy        : {accuracy * 100:.2f}%  ({n_correct}/{n_total})")
    print(f"Total inference time     : {elapsed:.2f} s")
    print(f"Inference throughput     : {throughput:.2f} images/s")
    print(f"MACs (per forward pass)  : {fm['MACs']:,}")
    print(f"FLOPs (per forward pass) : {fm['FLOPs']:,}")
    print(f"Trainable parameters     : {n_params:,}")
    print("=" * 55)


if __name__ == "__main__":
    main()
