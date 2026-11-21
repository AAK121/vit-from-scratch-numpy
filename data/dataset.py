"""ImageFolder-style dataset handling for the NumPy Vision Transformer.

Two code paths:
  - A disk-cache path (build_cache/load_cache/ArrayLoader) used for local training and the
    model/data-scaling sweeps: images are resized once and memmapped as uint8, so repeated
    epochs across many experiment configs don't re-decode PNGs every time.
  - An on-the-fly path (TestLoader) used by test.py against the hidden test set: no caching,
    reads+resizes directly with Pillow, since the hidden test directory is only known at
    evaluation time and is evaluated once.
"""
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from nn.backend import to_device

IMG_EXTS = (".png", ".jpg", ".jpeg")


def discover_classes(root):
    return sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )


def list_files(root, classes):
    items = []
    for label, c in enumerate(classes):
        cdir = os.path.join(root, c)
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith(IMG_EXTS):
                items.append((os.path.join(cdir, fname), label))
    return items


def _resize_and_crop(im, img_size):
    """Resize shorter side to img_size, center-crop to img_size x img_size, RGB uint8.
    `im` is an already-opened, already-RGB-converted PIL Image (so callers that need the same
    source image at more than one size only decode it once)."""
    w, h = im.size
    scale = img_size / min(w, h)
    nw, nh = max(img_size, round(w * scale)), max(img_size, round(h * scale))
    resized = im.resize((nw, nh), Image.BILINEAR)
    left, top = (nw - img_size) // 2, (nh - img_size) // 2
    resized = resized.crop((left, top, left + img_size, top + img_size))
    arr = np.asarray(resized, dtype=np.uint8)
    if arr.shape != (img_size, img_size, 3):
        out = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        h2, w2 = arr.shape[:2]
        out[: min(h2, img_size), : min(w2, img_size)] = arr[:img_size, :img_size]
        arr = out
    return arr


def load_and_resize(path, img_size):
    """Resize shorter side to img_size, center-crop to img_size x img_size, RGB uint8."""
    return _resize_and_crop(Image.open(path).convert("RGB"), img_size)


def center_crop_batch(imgs, crop_size):
    """imgs: (B, H, W, C) with H, W >= crop_size. Same crop position for every sample."""
    h, w = imgs.shape[1], imgs.shape[2]
    top, left = (h - crop_size) // 2, (w - crop_size) // 2
    return imgs[:, top:top + crop_size, left:left + crop_size, :]


def mixup_batch(imgs, labels, alpha, rng):
    """Mixup (Zhang et al., 2018): blends each image with a randomly-paired other image in the
    same batch, `lam*img_i + (1-lam)*img_perm[i]`, with a single batch-wide mixing coefficient
    `lam ~ Beta(alpha, alpha)`. This is purely a data-level transform (elementwise convex
    combination of already-loaded arrays) -- it introduces no new learnable layer and needs no
    hand-derived backward of its own: the loss it feeds into is just `lam * CE(logits, labels) +
    (1-lam) * CE(logits, labels_perm)`, a linear combination of two already-verified
    `softmax_cross_entropy` calls at the *same* logits, so `dlogits = lam*dlogits_a +
    (1-lam)*dlogits_b` is exact by linearity (verified directly in `scripts/gradcheck.py`'s
    `check_mixup_loss`). No-op (returns `imgs, labels, labels, 1.0`) when `alpha<=0`, matching
    every other augmentation flag in this codebase.

    Returns `(mixed_imgs, labels_a, labels_b, lam)`; the caller is responsible for combining the
    two cross-entropy losses/gradients with `lam`.
    """
    if alpha <= 0:
        return imgs, labels, labels, 1.0
    lam = float(rng.beta(alpha, alpha))
    perm = rng.permutation(imgs.shape[0])
    perm_dev = to_device(perm)
    mixed = lam * imgs + (1.0 - lam) * imgs[perm_dev]
    labels_b = labels[perm_dev]
    return mixed, labels, labels_b, lam


def random_crop_batch(imgs, crop_size, rng):
    """imgs: (B, H, W, C) with H, W >= crop_size. Independent random crop position per sample
    -- translation-jitter augmentation, see Sec. "Data distribution" / "fix the preprocessing":
    train.py caches images at a size larger than the model's input specifically so this can
    sample different, not-always-centered windows of each photo across epochs, rather than the
    same fixed center-crop (which can systematically cut off an off-center subject) every time.
    """
    b, h, w = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    max_top, max_left = h - crop_size, w - crop_size
    tops = rng.integers(0, max_top + 1, size=b)
    lefts = rng.integers(0, max_left + 1, size=b)
    out = np.empty((b, crop_size, crop_size, imgs.shape[3]), dtype=imgs.dtype)
    for i in range(b):
        out[i] = imgs[i, tops[i]:tops[i] + crop_size, lefts[i]:lefts[i] + crop_size, :]
    return out


# ---------------------------------------------------------------------------
# Disk-cache path (train/ side)
# ---------------------------------------------------------------------------

def _cache_paths(out_dir, img_size):
    return (
        os.path.join(out_dir, f"images_{img_size}.npy"),
        os.path.join(out_dir, f"labels_{img_size}.npy"),
        os.path.join(out_dir, f"meta_{img_size}.json"),
    )


def build_cache(root, out_dir, img_size, val_frac=0.1, seed=1234, force=False,
                 train_resize_size=None):
    """Builds the eval-resolution cache at img_size (used for val/pool bookkeeping, and kept
    byte-for-byte identical to before so validation accuracy and `test.py`'s on-the-fly
    TestLoader preprocessing stay in agreement). If `train_resize_size` is given (> img_size),
    also builds a second cache at that larger resolution, decoding each source image only once
    -- this is what train.py's train_loader samples random img_size windows from (see
    `random_crop_batch`), instead of always training on the exact same fixed center-crop.
    """
    os.makedirs(out_dir, exist_ok=True)
    imgs_path, labels_path, meta_path = _cache_paths(out_dir, img_size)
    build_big = train_resize_size is not None and train_resize_size > img_size
    if build_big:
        big_imgs_path, big_labels_path, big_meta_path = _cache_paths(out_dir, train_resize_size)

    already_built = os.path.exists(imgs_path) and os.path.exists(meta_path)
    if build_big:
        already_built = already_built and os.path.exists(big_imgs_path) and os.path.exists(big_meta_path)
    if already_built and not force:
        with open(meta_path) as f:
            return json.load(f)

    classes = discover_classes(root)
    items = list_files(root, classes)
    n = len(items)

    images = np.lib.format.open_memmap(
        imgs_path, mode="w+", dtype=np.uint8, shape=(n, img_size, img_size, 3)
    )
    if build_big:
        big_images = np.lib.format.open_memmap(
            big_imgs_path, mode="w+", dtype=np.uint8,
            shape=(n, train_resize_size, train_resize_size, 3),
        )
    labels = np.zeros((n,), dtype=np.int16)
    desc = f"caching @ {img_size}px" + (f" (+{train_resize_size}px for train aug.)" if build_big else "")
    for i, (path, label) in enumerate(tqdm(items, desc=desc)):
        im = Image.open(path).convert("RGB")
        images[i] = _resize_and_crop(im, img_size)
        if build_big:
            big_images[i] = _resize_and_crop(im, train_resize_size)
        labels[i] = label
    images.flush()
    np.save(labels_path, labels)
    if build_big:
        big_images.flush()
        np.save(big_labels_path, labels)

    rng = np.random.default_rng(seed)
    val_idx, pool_idx = [], []
    for c in range(len(classes)):
        idx_c = np.where(labels == c)[0].copy()
        rng.shuffle(idx_c)
        n_val = max(1, int(round(len(idx_c) * val_frac)))
        val_idx.extend(idx_c[:n_val].tolist())
        pool_idx.extend(idx_c[n_val:].tolist())
    val_idx = sorted(val_idx)
    pool_idx = sorted(pool_idx)

    sample_n = min(len(pool_idx), 20000)
    sample_idx = np.array(sorted(pool_idx[:sample_n]))
    sample = images[sample_idx].astype(np.float64) / 255.0
    mean = sample.mean(axis=(0, 1, 2)).tolist()
    std = sample.std(axis=(0, 1, 2)).tolist()

    meta = dict(
        classes=classes,
        img_size=img_size,
        n=n,
        val_idx=val_idx,
        pool_idx=pool_idx,
        mean=mean,
        std=std,
        seed=seed,
        val_frac=val_frac,
        train_resize_size=train_resize_size,
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    if build_big:
        # Same classes/split/mean/std/labels; only img_size differs (kept for load_cache symmetry).
        big_meta = dict(meta, img_size=train_resize_size)
        with open(big_meta_path, "w") as f:
            json.dump(big_meta, f)
    return meta


def load_cache(out_dir, img_size):
    imgs_path, labels_path, meta_path = _cache_paths(out_dir, img_size)
    images = np.load(imgs_path, mmap_mode="r")
    labels = np.load(labels_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return images, labels, meta


def subsample_uniform(pool_idx, labels, fraction, seed):
    """Stratified: draw the same fraction from every class."""
    rng = np.random.default_rng(seed)
    pool_idx = np.asarray(pool_idx)
    pool_labels = labels[pool_idx]
    chosen = []
    for c in np.unique(pool_labels):
        idx_c = pool_idx[pool_labels == c].copy()
        rng.shuffle(idx_c)
        k = max(1, int(round(len(idx_c) * fraction)))
        chosen.extend(idx_c[:k].tolist())
    return np.array(sorted(chosen))


def subsample_nonuniform(pool_idx, labels, fraction, seed, alpha=0.3):
    """Non-uniform: skewed per-class quotas drawn from a Dirichlet(alpha) distribution over
    class proportions (small alpha -> heavy class imbalance), rather than an equal share per
    class. Total sample count still targets len(pool) * fraction.
    """
    rng = np.random.default_rng(seed)
    pool_idx = np.asarray(pool_idx)
    pool_labels = labels[pool_idx]
    classes = np.unique(pool_labels)
    total = max(len(classes), int(round(len(pool_idx) * fraction)))
    props = rng.dirichlet(alpha * np.ones(len(classes)))
    chosen = []
    for c, p in zip(classes, props):
        avail = pool_idx[pool_labels == c].copy()
        rng.shuffle(avail)
        k = min(len(avail), max(0, int(round(p * total))))
        chosen.extend(avail[:k].tolist())
    return np.array(sorted(chosen))


class ArrayLoader:
    """Batches over a memmapped uint8 image cache + label array.

    `images` may be cached larger than `img_size` (e.g. train.py feeds this the
    `train_resize_size` cache for the training split): if so, each batch is cropped down to
    `img_size` -- a random window per sample when `augment=True` (translation-jitter
    augmentation, see `random_crop_batch`), else a fixed center crop (matches the eval/test
    preprocessing exactly). When `images` is already exactly `img_size`, this is a no-op, so
    the validation loader's output is unchanged from before this option existed.
    """

    def __init__(self, images, labels, indices, mean, std, batch_size=64,
                 shuffle=True, augment=False, seed=0, img_size=None):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.img_size = img_size if img_size is not None else images.shape[1]

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.indices.copy()
        if self.shuffle:
            self.rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            batch_idx = np.sort(idx[start:start + self.batch_size])
            raw = np.asarray(self.images[batch_idx])
            if raw.shape[1] > self.img_size:
                raw = random_crop_batch(raw, self.img_size, self.rng) if self.augment \
                    else center_crop_batch(raw, self.img_size)
            imgs = raw.astype(np.float32) / 255.0
            imgs = (imgs - self.mean) / self.std
            if self.augment:
                flip = self.rng.random(imgs.shape[0]) < 0.5
                imgs[flip] = imgs[flip, :, ::-1, :]
            labels = self.labels[batch_idx].astype(np.int64)
            yield to_device(imgs), to_device(labels)


# ---------------------------------------------------------------------------
# On-the-fly path (hidden test set side, used by test.py)
# ---------------------------------------------------------------------------

class TestLoader:
    """Loads a test set directly from disk (no caching), mapping class-dir names to the
    label indices the model was trained with (persisted in the checkpoint config).
    """

    def __init__(self, test_path, classes, img_size, mean, std, batch_size=64):
        class_to_idx = {c: i for i, c in enumerate(classes)}
        test_classes = discover_classes(test_path)
        unknown = [c for c in test_classes if c not in class_to_idx]
        if unknown:
            raise ValueError(f"Test set contains classes not seen in training: {unknown}")

        items = []
        for c in test_classes:
            label = class_to_idx[c]
            cdir = os.path.join(test_path, c)
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(IMG_EXTS):
                    items.append((os.path.join(cdir, fname), label))

        self.items = items
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for start in range(0, len(self.items), self.batch_size):
            chunk = self.items[start:start + self.batch_size]
            imgs = np.stack(
                [load_and_resize(p, self.img_size) for p, _ in chunk]
            ).astype(np.float32) / 255.0
            imgs = (imgs - self.mean) / self.std
            labels = np.array([l for _, l in chunk], dtype=np.int64)
            yield to_device(imgs), to_device(labels)
