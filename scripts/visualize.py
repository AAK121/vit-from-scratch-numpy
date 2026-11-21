import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from data.dataset import load_cache
from nn.backend import to_host, xp
from nn.transformer import build_vit
from utils.checkpoint import load_into_model

RESULTS_DIR = "results/plots"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/submission")
    p.add_argument("--cache-dir", type=str, default="data_cache")
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--n-correct", type=int, default=4)
    p.add_argument("--n-wrong", type=int, default=4)
    p.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "gradcam_examples.png"))
    p.add_argument("--scan-batch-size", type=int, default=256,
                    help="batch size for the initial full-val-set correct/wrong scan (not the "
                         "final small Grad-CAM/rollout visualization, which stays unbatched). "
                         "The whole validation set was previously forwarded in one shot, which "
                         "is fine at 64px but OOMs at larger resolutions/token counts (e.g. "
                         "256px, N=257) since attention score matrices scale with batch size "
                         "times N^2.")
    return p.parse_args()


def forward_to_last_block(model, x):
    """Replicates VisionTransformer.forward, but also returns `pre_first` -- the token
    sequence *entering the first* transformer block (patch + CLS + positional embeddings,
    before any attention mixing). This is the Grad-CAM target.

    The naive choice -- targeting the *last* block's output -- does not work for this
    checkpoint: the classifier head only ever reads the CLS row of ln_final(feat), and
    LayerNorm normalizes each token independently (no cross-token mixing), so patch tokens
    only reach the CLS logit *through* attention. Empirically, this checkpoint's last three
    blocks (1, 2, 3) have collapsed to *exactly* uniform attention (std 0.0, see
    `attention_rollout`'s docstring) -- a real over-smoothing/representation-collapse failure
    mode -- so backpropping through only the last block (or last two) leaves a gradient that is
    zero to within floating point (measured std ~1e-22 through block 3 alone, ~1e-7 through
    blocks 3+2). Backpropping through *all four* blocks recovers a real, non-degenerate signal
    (std ~3e-2) because block 0's attention is not collapsed and re-injects genuine spatial
    structure into the gradient on the way out. So Grad-CAM here uses the full backward pass
    (head -> ln_final -> block3 -> block2 -> block1 -> block0), exactly the same backward()
    chain VisionTransformer.backward runs during training, just stopped one step earlier
    (before patch_embed's backward, which is a pure reshape and has no "features" to explain).
    """
    h = model.patch_embed.forward(x, training=False)
    h = model.cls_pos.forward(h, training=False)
    pre_first = h  # (B, N, D) -- input to the first transformer block
    for blk in model.blocks:
        h = blk.forward(h, training=False)
    h2 = model.ln_final.forward(h, training=False)
    cls = h2[:, 0, :]
    logits = model.head.forward(cls, training=False)
    return pre_first, h2, logits


def grad_cam(model, pre_first, h2, target_idx):
    """Backprop the target-class logit through head -> ln_final -> every transformer block
    (in reverse) down to `pre_first`, using only the model's own hand-written backward methods
    -- exactly the Core Component 1 backward pass, reused for interpretability rather than
    optimization. Then apply standard Grad-CAM pooling: GAP the gradient over patches to get
    per-channel weights, weight the activations, sum over channels, ReLU.

    All intermediate math uses `xp` (numpy or cupy, whichever the model's own forward/backward
    used) rather than plain `numpy`, since the model's backward() methods expect device-
    consistent arrays throughout; only the final returned `cam` is moved to host (numpy), since
    everything downstream of this function (PIL-based upsampling/plotting) is host-only.
    """
    B, N, D = h2.shape
    num_classes = model.head.out_dim
    target_idx = xp.asarray(target_idx)
    dlogits = xp.zeros((B, num_classes), dtype=xp.float32)
    dlogits[xp.arange(B), target_idx] = 1.0

    dcls = model.head.backward(dlogits)             # (B, D)
    dx = xp.zeros_like(h2)
    dx[:, 0, :] = dcls
    g = model.ln_final.backward(dx)                  # (B, N, D)
    for blk in reversed(model.blocks):
        g = blk.backward(g)                          # mixes across tokens via attention
    dpre_first = g                                    # (B, N, D) dLogit/dPreFirst

    patch_act = pre_first[:, 1:, :]                   # (B, P, D) drop CLS
    patch_grad = dpre_first[:, 1:, :]
    weights = patch_grad.mean(axis=1)                 # (B, D) global-average-pool over patches
    cam = xp.maximum((weights[:, None, :] * patch_act).sum(axis=-1), 0.0)  # (B, P)
    cam = cam / (cam.max(axis=1, keepdims=True) + 1e-8)
    return to_host(cam)


def attention_rollout(model, discard_ratio=0.0):
    """Attention Rollout (Abnar & Zuidema, 2020): multiply every block's (head-averaged, +
    identity for the residual stream, row-renormalized) attention matrix together across depth,
    then read off the CLS row. Unlike a single block's raw attention (which can saturate to
    near-uniform in deeper blocks -- observed in this checkpoint for blocks 1-3), rollout
    accumulates the *whole* attention pathway from input patches to the CLS token, which is the
    standard "best for ViT" attention-based saliency method.

    Uses `xp` throughout (see `grad_cam`'s docstring) since the cached attention weights live on
    whichever device the forward pass ran on; only the final `cls_row` is moved to host.
    """
    B = None
    result = None
    for blk in model.blocks:
        attn = blk.attn._cache[3]  # (B, H, N, N)
        if B is None:
            B = attn.shape[0]
            N = attn.shape[-1]
            result = xp.broadcast_to(xp.eye(N, dtype=xp.float32), (B, N, N)).copy()
        a = attn.mean(axis=1)  # (B, N, N) average over heads
        a = 0.5 * a + 0.5 * xp.eye(a.shape[-1], dtype=xp.float32)[None]
        a = a / a.sum(axis=-1, keepdims=True)
        result = xp.matmul(a, result)
    cls_row = result[:, 0, 1:]  # (B, P) CLS row, patches only
    cls_row = cls_row / (cls_row.max(axis=1, keepdims=True) + 1e-8)
    return to_host(cls_row)


def upsample_map(m, grid, img_size):
    m = m.reshape(grid, grid)
    im = Image.fromarray((m * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def overlay(image_uint8, heat, cmap_name="jet", alpha=0.45):
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    heat_rgb = (cmap(heat)[..., :3] * 255).astype(np.float32)
    base = image_uint8.astype(np.float32)
    blended = (1 - alpha) * base + alpha * heat_rgb
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    with open(args.checkpoint + ".json") as f:
        ckpt_meta = json.load(f)
    config = ckpt_meta["config"]
    extra = ckpt_meta["extra"]
    classes, mean, std = extra["classes"], extra["mean"], extra["std"]

    model = build_vit(config)
    load_into_model(model, args.checkpoint)

    images, labels, meta = load_cache(args.cache_dir, args.img_size)
    val_idx = np.array(meta["val_idx"])
    mean_a = np.array(mean, dtype=np.float32).reshape(1, 1, 1, 3)
    std_a = np.array(std, dtype=np.float32).reshape(1, 1, 1, 3)

    batch_imgs = np.asarray(images[val_idx], dtype=np.float32) / 255.0
    batch_imgs = (batch_imgs - mean_a) / std_a
    batch_labels = labels[val_idx]

    # Chunked scan (not one giant forward pass): attention score matrices scale with
    # batch_size * N^2, so forwarding all 5,000 validation images at once is fine at 64px
    # (N=65) but can OOM at larger resolutions (e.g. 256px, N=257).
    full_preds = np.empty(len(batch_imgs), dtype=np.int64)
    bs = args.scan_batch_size
    for start in range(0, len(batch_imgs), bs):
        _, _, chunk_logits = forward_to_last_block(model, batch_imgs[start:start + bs])
        full_preds[start:start + bs] = to_host(chunk_logits.argmax(axis=-1))
    correct_mask = full_preds == batch_labels

    correct_pool = np.where(correct_mask)[0]
    wrong_pool = np.where(~correct_mask)[0]
    chosen = np.concatenate([
        correct_pool[: args.n_correct],
        wrong_pool[: args.n_wrong],
    ])
    print(f"val accuracy on this run: {correct_mask.mean() * 100:.2f}% "
          f"({correct_mask.sum()}/{len(correct_mask)})")
    print(f"visualizing {len(chosen)} examples: "
          f"{min(args.n_correct, len(correct_pool))} correct, "
          f"{min(args.n_wrong, len(wrong_pool))} misclassified")

    # Re-run forward on just the chosen subset so Grad-CAM's backward pass (and the cached
    # attention weights attention_rollout reads) line up with this smaller batch.
    pre_first, h2, logits = forward_to_last_block(model, batch_imgs[chosen])
    preds = logits.argmax(axis=-1)
    cams = grad_cam(model, pre_first, h2, preds)
    preds = to_host(preds)  # host from here on: plain-Python indexing (classes[preds[r]]) below
    attn_maps = attention_rollout(model)

    grid = model.patch_embed.grid
    img_size = args.img_size

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(chosen)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Original", "Grad-CAM (full hand-written backward)", "Attention Rollout"]
    for r, idx in enumerate(chosen):
        raw = np.asarray(images[val_idx[idx]])  # uint8 HWC, un-normalized, for display
        cam_up = upsample_map(cams[r], grid, img_size)
        attn_up = upsample_map(attn_maps[r], grid, img_size)

        true_c, pred_c = classes[batch_labels[idx]], classes[preds[r]]
        tag = "correct" if correct_mask[idx] else "WRONG"
        axes[r, 0].imshow(raw)
        axes[r, 0].set_ylabel(f"true={true_c}\npred={pred_c} ({tag})", fontsize=7)
        axes[r, 1].imshow(overlay(raw, cam_up))
        axes[r, 2].imshow(overlay(raw, attn_up))
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c in range(3):
                axes[r, c].set_title(col_titles[c], fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
