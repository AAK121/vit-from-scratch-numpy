import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["VIT_DEVICE"] = "cpu"  # np.nditer (finite differences) needs a real NumPy array

import numpy as np

from nn.layers import (
    Linear, LayerNorm, GELU, Dropout, PatchEmbedding, ClsPosEmbed, Conv2d, DepthwiseConv2d,
    ConvPatchStem,
)
from nn.dct import DCT2D, DCTTruncate
from nn.attention import MultiHeadSelfAttention
from nn.transformer import TransformerBlock, VisionTransformer
from nn.losses import softmax_cross_entropy

EPS = 1e-3
TOL = 5e-2  # generous relative-error tolerance (float32 params, finite differences)


def rel_error(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def numeric_grad(loss_fn, arr, eps=EPS):
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = arr[idx]
        arr[idx] = orig + eps
        fp = loss_fn()
        arr[idx] = orig - eps
        fm = loss_fn()
        arr[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
    return grad


def check_layer(name, layer, x, results):
    out = layer.forward(x, training=True)
    rng = np.random.default_rng(0)
    dout = rng.standard_normal(out.shape)

    def loss_fn():
        o = layer.forward(x, training=True)
        return float(np.sum(o * dout))

    dx_analytic = layer.backward(dout)
    grads_analytic = dict(layer.named_grads())

    dx_numeric = numeric_grad(loss_fn, x)
    err_x = rel_error(dx_analytic, dx_numeric)
    results.append((f"{name}.dx", err_x))
    print(f"  dx        rel err = {err_x:.2e}")

    for pname, p in layer.named_parameters():
        g_num = numeric_grad(loss_fn, p)
        g_an = grads_analytic[pname]
        err = rel_error(g_an, g_num)
        results.append((f"{name}.{pname}", err))
        print(f"  d{pname:10s} rel err = {err:.2e}")


def check_dropout(results):
    """Dropout's forward draws a fresh random mask every call, so it can't go through
    check_layer's generic loss_fn (which calls layer.forward() again per finite-difference
    probe -- a *different* mask each time, which would corrupt the numeric gradient). Instead
    the mask is drawn once and reused directly to build loss_fn, isolating the check to
    "does backward correctly apply that fixed mask" (the actual hand-derived logic), which is
    exactly what matters since the mask itself is just a Bernoulli draw with no gradient.
    """
    print("\n[Dropout] (training=True, p=0.4, fixed mask reused across finite-difference probes)")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 5, 6))
    drop = Dropout(0.4)
    out = drop.forward(x, training=True)
    mask = drop._mask
    dout = rng.standard_normal(out.shape)

    def loss_fn():
        return float(np.sum((x * mask) * dout))

    dx_analytic = drop.backward(dout)
    dx_numeric = numeric_grad(loss_fn, x)
    err = rel_error(dx_analytic, dx_numeric)
    results.append(("Dropout.dx", err))
    print(f"  dx        rel err = {err:.2e}")

    print("[Dropout] (training=False / p=0 -> identity)")
    assert np.array_equal(Dropout(0.4).forward(x, training=False), x), "eval-mode dropout must be identity"
    assert np.array_equal(Dropout(0.0).forward(x, training=True), x), "p=0 dropout must be identity"
    print("  OK: eval-mode and p=0 are both exact no-ops")


def check_label_smoothing(results):
    print("\n[softmax_cross_entropy] (label_smoothing=0.1)")
    rng = np.random.default_rng(3)
    B, K = 4, 6
    logits = rng.standard_normal((B, K))
    labels = rng.integers(0, K, size=(B,))

    def loss_fn():
        loss, _, _ = softmax_cross_entropy(logits, labels, label_smoothing=0.1)
        return loss

    loss, dlogits_analytic, _ = softmax_cross_entropy(logits, labels, label_smoothing=0.1)
    dlogits_numeric = numeric_grad(loss_fn, logits)
    err = rel_error(dlogits_analytic, dlogits_numeric)
    results.append(("softmax_cross_entropy(label_smoothing=0.1).dlogits", err))
    print(f"  dlogits   rel err = {err:.2e}")


def check_mixup_loss(results):
    """Mixup (data.dataset.mixup_batch) introduces no new layer -- it mixes inputs before the
    model and combines two already-verified softmax_cross_entropy calls linearly. Checked here
    directly: dlogits = lam*dlogits_a + (1-lam)*dlogits_b should match the numeric gradient of
    lam*CE(logits,labels_a) + (1-lam)*CE(logits,labels_b) w.r.t. logits exactly (up to finite-
    difference tolerance), by linearity of differentiation."""
    print("\n[mixup loss] (lam*CE(logits,labels_a) + (1-lam)*CE(logits,labels_b))")
    rng = np.random.default_rng(4)
    B, K = 5, 6
    logits = rng.standard_normal((B, K))
    labels_a = rng.integers(0, K, size=(B,))
    labels_b = rng.integers(0, K, size=(B,))
    lam = 0.37

    def loss_fn():
        la, _, _ = softmax_cross_entropy(logits, labels_a)
        lb, _, _ = softmax_cross_entropy(logits, labels_b)
        return lam * la + (1.0 - lam) * lb

    _, dlogits_a, _ = softmax_cross_entropy(logits, labels_a)
    _, dlogits_b, _ = softmax_cross_entropy(logits, labels_b)
    dlogits_analytic = lam * dlogits_a + (1.0 - lam) * dlogits_b
    dlogits_numeric = numeric_grad(loss_fn, logits)
    err = rel_error(dlogits_analytic, dlogits_numeric)
    results.append(("mixup_loss.dlogits", err))
    print(f"  dlogits   rel err = {err:.2e}")


def check_full_model(results, conv_stem=False, local_conv=False, tokenizer="raw", dct_keep=None,
                      label=None):
    print(f"\n[{label or 'VisionTransformer + softmax_cross_entropy'}] (end-to-end)")
    rng = np.random.default_rng(1)
    B, img, patch, chans, ncls = 2, 8, 4, 2, 3
    model = VisionTransformer(
        img_size=img,
        patch_size=patch,
        in_chans=chans,
        num_classes=ncls,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        seed=42,
        conv_stem=conv_stem,
        local_conv=local_conv,
        tokenizer=tokenizer,
        dct_keep=dct_keep,
    )
    x = rng.standard_normal((B, img, img, chans))
    labels = rng.integers(0, ncls, size=(B,))

    def loss_fn():
        logits = model.forward(x, training=True)
        loss, _, _ = softmax_cross_entropy(logits, labels)
        return loss

    logits = model.forward(x, training=True)
    loss, dlogits, _ = softmax_cross_entropy(logits, labels)
    dx_analytic = model.backward(dlogits)
    grads_analytic = dict(model.named_grads())

    dx_numeric = numeric_grad(loss_fn, x)
    err_x = rel_error(dx_analytic, dx_numeric)
    results.append(("model.dx", err_x))
    print(f"  dx (input)              rel err = {err_x:.2e}")

    for pname, p in model.named_parameters():
        g_num = numeric_grad(loss_fn, p)
        g_an = grads_analytic[pname]
        err = rel_error(g_an, g_num)
        results.append((f"model.{pname}", err))
        status = "OK" if err < TOL else "FAIL"
        print(f"  d{pname:28s} rel err = {err:.2e}  [{status}]")


def main():
    rng = np.random.default_rng(0)
    results = []

    print("[Linear]")
    x = rng.standard_normal((3, 5, 4))
    check_layer("Linear", Linear(4, 6, rng=rng), x, results)

    print("\n[LayerNorm]")
    x = rng.standard_normal((3, 5, 6))
    check_layer("LayerNorm", LayerNorm(6), x, results)

    print("\n[GELU]")
    x = rng.standard_normal((3, 5, 6))
    check_layer("GELU", GELU(), x, results)

    print("\n[MultiHeadSelfAttention]")
    x = rng.standard_normal((2, 3, 8))
    check_layer("MHSA", MultiHeadSelfAttention(8, 2, rng=rng), x, results)

    print("\n[PatchEmbedding]")
    x = rng.standard_normal((2, 8, 8, 2))
    check_layer("PatchEmbedding", PatchEmbedding(8, 4, 2, 6, rng=rng), x, results)

    print("\n[ClsPosEmbed]")
    x = rng.standard_normal((2, 4, 6))
    check_layer("ClsPosEmbed", ClsPosEmbed(4, 6, rng=rng), x, results)

    print("\n[Conv2d] (stride 2, padding 1, 3x3 -- ConvPatchStem's building block)")
    x = rng.standard_normal((2, 8, 8, 3))
    check_layer("Conv2d", Conv2d(3, 5, 3, stride=2, padding=1, rng=rng), x, results)

    print("\n[DepthwiseConv2d] (3x3, 'same' padding -- TransformerBlock's local_conv unit)")
    x = rng.standard_normal((2, 4, 4, 6))
    check_layer("DepthwiseConv2d", DepthwiseConv2d(6, kernel_size=3, rng=rng), x, results)

    print("\n[ConvPatchStem] (patch_size=8 -> 3 stride-2 convs)")
    x = rng.standard_normal((2, 8, 8, 2))
    check_layer("ConvPatchStem", ConvPatchStem(8, 8, 2, 6, rng=rng), x, results)

    print("\n[DCT2D] (ported from token/, fixed orthonormal transform -- no learnable params)")
    x = rng.standard_normal((2, 3, 4, 4, 2))  # B, N, P, P, C
    check_layer("DCT2D", DCT2D(4), x, results)

    print("\n[DCTTruncate] (fixed 0/1 selection matrix -- no learnable params)")
    x = rng.standard_normal((2, 3, 4, 4, 2))  # B, N, P, P, C
    check_layer("DCTTruncate", DCTTruncate(4, keep_k=6), x, results)

    print("\n[PatchEmbedding, tokenizer=dct, dct_keep=None (all coefficients)]")
    x = rng.standard_normal((2, 8, 8, 2))
    check_layer("PatchEmbedding[dct-full]",
                PatchEmbedding(8, 4, 2, 6, rng=rng, tokenizer="dct"), x, results)

    print("\n[PatchEmbedding, tokenizer=dct, dct_keep=6 (truncated)]")
    x = rng.standard_normal((2, 8, 8, 2))
    check_layer("PatchEmbedding[dct-trunc]",
                PatchEmbedding(8, 4, 2, 6, rng=rng, tokenizer="dct", dct_keep=6), x, results)

    print("\n[TransformerBlock]")
    x = rng.standard_normal((2, 3, 8))
    check_layer("TransformerBlock", TransformerBlock(8, 2, 16, rng=rng), x, results)

    print("\n[TransformerBlock] (local_conv_grid=2, i.e. local_conv=True on a 2x2 patch grid)")
    x = rng.standard_normal((2, 5, 8))  # 1 CLS + 4 spatial tokens (2x2 grid)
    check_layer("TransformerBlock+local_conv",
                TransformerBlock(8, 2, 16, rng=rng, local_conv_grid=2), x, results)

    check_dropout(results)
    check_label_smoothing(results)
    check_mixup_loss(results)
    check_full_model(results)
    check_full_model(results, conv_stem=True, local_conv=True,
                      label="VisionTransformer(conv_stem=True, local_conv=True) + softmax_cross_entropy")
    check_full_model(results, tokenizer="dct", dct_keep=None,
                      label="VisionTransformer(tokenizer=dct, dct_keep=None) + softmax_cross_entropy")
    check_full_model(results, tokenizer="dct", dct_keep=6,
                      label="VisionTransformer(tokenizer=dct, dct_keep=6) + softmax_cross_entropy")

    print("\n" + "=" * 60)
    n_fail = sum(1 for _, e in results if e >= TOL or np.isnan(e))
    worst = max(results, key=lambda kv: kv[1])
    print(f"Checked {len(results)} gradients. Worst: {worst[0]} = {worst[1]:.2e}")
    if n_fail:
        print(f"FAILED: {n_fail} gradient(s) exceeded tolerance {TOL:.0e}")
        for name, err in results:
            if err >= TOL or np.isnan(err):
                print(f"  FAIL {name}: {err:.2e}")
        sys.exit(1)
    else:
        print(f"PASSED: all gradients within tolerance {TOL:.0e}")


if __name__ == "__main__":
    main()
