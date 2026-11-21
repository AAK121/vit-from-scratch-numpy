"""Tiny end-to-end sanity check: can the assembled VisionTransformer overfit a handful of
images? This exercises the real data path (PIL loading/resizing), the real model, the real
optimizer, and the real training loop -- without waiting on a full 50k-image cache build.

Run: python scripts/smoke_test.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.dataset import discover_classes, list_files, load_and_resize
from nn.backend import DEVICE, to_device
from nn.losses import softmax_cross_entropy
from nn.optim import Adam
from nn.transformer import build_vit

N_CLASSES = 5
N_PER_CLASS = 8
IMG_SIZE = 64
EPOCHS = 60


def main():
    root = "train"
    classes = discover_classes(root)[:N_CLASSES]
    items = list_files(root, classes)
    by_class = {}
    for path, label in items:
        by_class.setdefault(label, []).append(path)

    paths, labels = [], []
    for label, plist in by_class.items():
        for p in plist[:N_PER_CLASS]:
            paths.append(p)
            labels.append(label)

    print(f"Loading {len(paths)} images from {N_CLASSES} classes...")
    imgs = np.stack([load_and_resize(p, IMG_SIZE) for p in paths]).astype(np.float32) / 255.0
    mean = imgs.mean(axis=(0, 1, 2), keepdims=True)
    std = imgs.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    imgs = (imgs - mean) / std
    labels = np.array(labels, dtype=np.int64)
    imgs = to_device(imgs)
    labels = to_device(labels)
    print(f"Backend device: {DEVICE}")

    config = dict(img_size=IMG_SIZE, patch_size=8, in_chans=3, num_classes=N_CLASSES,
                  embed_dim=64, depth=2, num_heads=2, mlp_ratio=4.0, seed=0)
    model = build_vit(config)
    optim = Adam(model, lr=1e-3, weight_decay=0.0)

    print(f"Model params: {model.num_parameters():,}")
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        logits = model.forward(imgs, training=True)
        loss, dlogits, _ = softmax_cross_entropy(logits, labels)
        model.backward(dlogits)
        optim.step()
        optim.zero_grad()
        acc = float((logits.argmax(axis=-1) == labels).mean())
        loss = float(loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss={loss:.4f}  train_acc={acc:.3f}")
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s. Final train_acc={acc:.3f}")
    if acc > 0.95:
        print("PASS: model can overfit a tiny dataset.")
    else:
        print("FAIL: model did not reach >95% train accuracy on a tiny dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
