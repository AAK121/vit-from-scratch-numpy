"""
  python train.py --config configs/base.json --tag baseline
  python train.py --config configs/base.json --tag data10_uniform --data-fraction 0.1 --sampling uniform
"""
import argparse
import json
import math
import os
import time

import numpy as np

from data.dataset import (ArrayLoader, build_cache, load_cache, mixup_batch, subsample_nonuniform,
                          subsample_uniform)
from nn.losses import softmax_cross_entropy
from nn.optim import Adam
from nn.transformer import build_vit
from utils.checkpoint import save_checkpoint
from utils.flops import count_flops_macs
from utils.logging_utils import CSVLogger
from utils.metrics import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/base.json")
    p.add_argument("--tag", type=str, required=True, help="run name; outputs go to checkpoints/<tag> and results/<tag>.csv")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None,
                    help="residual-branch dropout prob (attn output + MLP output + "
                         "embedding), applied inside every TransformerBlock; 0 (default) "
                         "is a no-op, matching every pre-existing checkpoint/gradcheck")
    p.add_argument("--label-smoothing", type=float, default=None,
                    help="softmax cross-entropy label smoothing eps (e.g. 0.1); 0 (default) "
                         "is plain one-hot cross-entropy")
    p.add_argument("--warmup-epochs", type=int, default=None,
                    help="linear LR warmup from 0 to --lr over this many epochs before "
                         "--lr-schedule takes over (or lr stays at --lr if schedule=none); "
                         "0 (default) disables warmup")
    p.add_argument("--mixup-alpha", type=float, default=None,
                    help="Mixup Beta(alpha, alpha) coefficient (e.g. 0.2); 0 (default) disables "
                         "Mixup. See data.dataset.mixup_batch")
    p.add_argument("--train-resize-size", type=int, default=None,
                    help="override configs' train_resize_size; pass 0 to disable train-time "
                         "random-crop augmentation (train directly on the img_size cache)")
    p.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"],
                    help="'cosine' decays lr from --lr down to ~0 across the --epochs budget "
                         "(not the early-stopped length, since that's not known in advance)")
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "nonuniform"])
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch (debug/smoke use)")
    p.add_argument("--early-stop-patience", type=int, default=15,
                    help="stop if val_acc doesn't improve (by --early-stop-min-delta) for this "
                         "many consecutive epochs; 0 disables early stopping")
    p.add_argument("--early-stop-min-delta", type=float, default=0.0,
                    help="minimum val_acc improvement over the current best to reset patience")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = json.load(f)

    for k, argk in [("epochs", "epochs"), ("batch_size", "batch_size"), ("lr", "lr"),
                     ("embed_dim", "embed_dim"), ("depth", "depth"), ("num_heads", "num_heads"),
                     ("patch_size", "patch_size"), ("train_resize_size", "train_resize_size"),
                     ("weight_decay", "weight_decay"), ("dropout", "dropout"),
                     ("label_smoothing", "label_smoothing"), ("warmup_epochs", "warmup_epochs"),
                     ("mixup_alpha", "mixup_alpha")]:
        v = getattr(args, argk)
        if v is not None:
            config[k] = v

    train_resize_size = config.get("train_resize_size")
    build_cache(config["data_root"], config["cache_dir"], config["img_size"],
                val_frac=config["val_frac"], seed=config["cache_seed"],
                train_resize_size=train_resize_size)
    images, labels, meta = load_cache(config["cache_dir"], config["img_size"])
    config["num_classes"] = len(meta["classes"])

    if train_resize_size:
        # Larger cache for train-time random-crop augmentation (see ArrayLoader/random_crop_batch);
        # val_loader below deliberately keeps using the exact-img_size `images` array (unchanged
        # eval preprocessing, matching test.py's TestLoader byte-for-byte).
        train_images, _, _ = load_cache(config["cache_dir"], train_resize_size)
    else:
        train_images = images

    if args.data_fraction >= 0.999:
        train_idx = np.array(meta["pool_idx"])
    elif args.sampling == "uniform":
        train_idx = subsample_uniform(meta["pool_idx"], labels, args.data_fraction, args.sample_seed)
    else:
        train_idx = subsample_nonuniform(meta["pool_idx"], labels, args.data_fraction, args.sample_seed)

    train_loader = ArrayLoader(train_images, labels, train_idx, meta["mean"], meta["std"],
                                batch_size=config["batch_size"], shuffle=True, augment=True,
                                seed=config["seed"], img_size=config["img_size"])
    val_loader = ArrayLoader(images, labels, meta["val_idx"], meta["mean"], meta["std"],
                              batch_size=config["batch_size"], shuffle=False, augment=False)

    model = build_vit(config)
    optim = Adam(model, lr=config["lr"], weight_decay=config.get("weight_decay", 0.0))
    label_smoothing = config.get("label_smoothing", 0.0)
    mixup_alpha = config.get("mixup_alpha", 0.0)
    mixup_rng = np.random.default_rng(config["seed"] + 1)

    n_params = model.num_parameters()
    fm = count_flops_macs(config)
    print(f"[{args.tag}] params={n_params:,} MACs={fm['MACs']:,} FLOPs={fm['FLOPs']:,} "
          f"train_n={len(train_idx)} val_n={len(meta['val_idx'])}")

    out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    logger = CSVLogger(
        os.path.join(args.results_dir, f"{args.tag}.csv"),
        fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_s", "lr"],
    )

    base_lr = config["lr"]
    total_epochs = config["epochs"]
    warmup_epochs = config.get("warmup_epochs", 0)

    def lr_at(epoch):
        # Linear warmup 0 -> base_lr over the first `warmup_epochs` epochs (standard
        # transformer practice: Q/K/V and attention are poorly calibrated at random init, so
        # a full-size step early on can destabilize training / feed the attention-collapse
        # failure mode noted in the report). --lr-schedule (cosine/none) takes over for the
        # remaining epochs, decaying across the post-warmup budget rather than the full one.
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            return base_lr * epoch / warmup_epochs
        if args.lr_schedule == "cosine":
            denom = max(1, total_epochs - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / denom)
            return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
        return base_lr

    best_val_acc = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, config["epochs"] + 1):
        optim.set_lr(lr_at(epoch))

        t0 = time.perf_counter()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for bi, (imgs, targets) in enumerate(train_loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            if mixup_alpha > 0:
                # lam*CE(labels_a) + (1-lam)*CE(labels_b); gradient is exact by linearity
                # (checked in scripts/gradcheck.py::check_mixup_loss).
                imgs, targets_a, targets_b, lam = mixup_batch(imgs, targets, mixup_alpha, mixup_rng)
                logits = model.forward(imgs, training=True)
                loss_a, dlogits_a, _ = softmax_cross_entropy(logits, targets_a, label_smoothing=label_smoothing)
                loss_b, dlogits_b, _ = softmax_cross_entropy(logits, targets_b, label_smoothing=label_smoothing)
                loss = lam * loss_a + (1.0 - lam) * loss_b
                dlogits = lam * dlogits_a + (1.0 - lam) * dlogits_b
            else:
                logits = model.forward(imgs, training=True)
                loss, dlogits, _ = softmax_cross_entropy(logits, targets, label_smoothing=label_smoothing)
            model.backward(dlogits)
            optim.step()
            optim.zero_grad()

            run_loss += float(loss) * len(targets)
            run_correct += int((logits.argmax(axis=-1) == targets).sum())
            run_n += len(targets)

        train_loss = run_loss / max(1, run_n)
        train_acc = run_correct / max(1, run_n)
        val = evaluate(model, val_loader)
        epoch_time = time.perf_counter() - t0

        print(f"[{args.tag}] epoch {epoch}/{config['epochs']} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
              f"lr={optim.lr:.2e} ({epoch_time:.1f}s)")

        logger.log(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                         val_loss=val["loss"], val_acc=val["accuracy"], epoch_time_s=epoch_time,
                         lr=optim.lr))

        ckpt_extra = dict(epoch=epoch, val_acc=val["accuracy"], classes=meta["classes"],
                           mean=meta["mean"], std=meta["std"], train_n=len(train_idx),
                           data_fraction=args.data_fraction, sampling=args.sampling)
        save_checkpoint(model, os.path.join(out_dir, "last"), extra=ckpt_extra)
        if val["accuracy"] > best_val_acc + args.early_stop_min_delta:
            best_val_acc = val["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(model, os.path.join(out_dir, "best"), extra=ckpt_extra)
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"[{args.tag}] early stopping: val_acc has not improved by >= "
                      f"{args.early_stop_min_delta} for {epochs_without_improvement} epochs "
                      f"(best={best_val_acc:.4f}); stopping after epoch {epoch}/{config['epochs']}.")
                break

    logger.close()
    print(f"[{args.tag}] done. best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
