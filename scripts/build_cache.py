import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.json")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    meta = build_cache(
        config["data_root"], config["cache_dir"], config["img_size"],
        val_frac=config["val_frac"], seed=config["cache_seed"], force=args.force,
        train_resize_size=config.get("train_resize_size"),
    )
    print(f"classes={len(meta['classes'])} n={meta['n']} "
          f"train_pool={len(meta['pool_idx'])} val={len(meta['val_idx'])}")
    print(f"mean={meta['mean']} std={meta['std']}")


if __name__ == "__main__":
    main()
