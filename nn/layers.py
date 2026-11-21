import math as _math

from nn.backend import xp as np
from nn.module import Module
from nn import init
from nn.dct import DCT2D, DCTTruncate


class Linear(Module):
    """y = x @ W + b, applied over the last dimension of x (any leading batch dims)."""

    def __init__(self, in_dim, out_dim, rng=None, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = bias
        self.add_param("W", init.xavier_uniform((in_dim, out_dim), rng=rng))
        if bias:
            self.add_param("b", init.zeros((out_dim,)))
        self._x = None

    def forward(self, x, training=True):
        self._x = x
        out = x @ self._params["W"]
        if self.use_bias:
            out = out + self._params["b"]
        return out

    def backward(self, dout):
        x = self._x
        x2 = x.reshape(-1, self.in_dim)
        dout2 = dout.reshape(-1, self.out_dim)
        self.set_grad("W", x2.T @ dout2)
        if self.use_bias:
            self.set_grad("b", dout2.sum(axis=0))
        dx = dout2 @ self._params["W"].T
        return dx.reshape(x.shape)


class LayerNorm(Module):
    """Standard LayerNorm over the last dimension."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.add_param("gamma", init.ones((dim,)))
        self.add_param("beta", init.zeros((dim,)))
        self._cache = None

    def forward(self, x, training=True):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        xhat = (x - mu) / std
        out = self._params["gamma"] * xhat + self._params["beta"]
        self._cache = (xhat, std)
        return out

    def backward(self, dout):
        xhat, std = self._cache
        gamma = self._params["gamma"]
        N = self.dim
        axes = tuple(range(dout.ndim - 1))
        self.set_grad("gamma", (dout * xhat).sum(axis=axes))
        self.set_grad("beta", dout.sum(axis=axes))
        dxhat = dout * gamma
        dx = (1.0 / (N * std)) * (
            N * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx


class GELU(Module):
    """GELU, tanh approximation (as used in GPT-2/BERT): matches the exact erf-based GELU
    to within ~1e-3 but avoids scipy.special.erf, which profiled ~6-9x slower than np.tanh
    for the array sizes here -- a meaningful fraction of total training time. No learnable
    params.
    """

    _C = _math.sqrt(2.0 / _math.pi)  # plain Python float: avoids backend-dependent scalar ops
    _A = 0.044715

    def __init__(self):
        super().__init__()
        self._cache = None

    def forward(self, x, training=True):
        u = self._C * (x + self._A * x ** 3)
        t = np.tanh(u)
        self._cache = (x, t)
        return 0.5 * x * (1.0 + t)

    def backward(self, dout):
        x, t = self._cache
        du_dx = self._C * (1.0 + 3.0 * self._A * x * x)
        dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
        return dout * dgelu


class Dropout(Module):
    """Inverted dropout: zeroes each element independently with probability `p` and rescales
    the survivors by 1/(1-p) at train time, so the expected activation is unchanged and no
    rescaling is needed at eval time. No-op (identity, no RNG draw) when `p <= 0` or
    `training=False` -- this keeps `scripts/gradcheck.py`'s existing finite-difference checks
    deterministic for every call site that doesn't explicitly pass `dropout > 0`.
    """

    def __init__(self, p=0.0):
        super().__init__()
        assert 0.0 <= p < 1.0, "dropout probability must be in [0, 1)"
        self.p = p
        self._mask = None

    def forward(self, x, training=True):
        if not training or self.p <= 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        mask = (np.random.random(x.shape) < keep).astype(x.dtype) / keep
        self._mask = mask
        return x * mask

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask


class Conv2d(Module):
    """Standard (cross-channel) 2D convolution, NHWC layout, square kernel, symmetric padding.

    Implemented as a sum of `kernel_size**2` shifted/strided slices each matmul'd against that
    kernel tap's (in_chans, out_chans) weight slab -- avoids materializing an explicit im2col
    buffer while staying a handful of large batched matmuls (backend-agnostic: numpy or cupy),
    which is efficient for the small (3x3) kernels used here. Used by `ConvPatchStem` (TinyViT-
    style convolutional patchify, see that class) as a from-scratch alternative to
    `PatchEmbedding`'s single strided-linear patchify.
    """

    def __init__(self, in_chans, out_chans, kernel_size, stride=1, padding=0, rng=None,
                 bias=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.add_param("W", init.conv_xavier_uniform(
            (kernel_size, kernel_size, in_chans, out_chans), rng=rng))
        if bias:
            self.add_param("b", init.zeros((out_chans,)))
        self._cache = None

    def _out_hw(self, H, W):
        k, s, p = self.k, self.stride, self.padding
        return (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1

    def forward(self, x, training=True):
        B, H, W, Cin = x.shape
        k, s, p = self.k, self.stride, self.padding
        Hout, Wout = self._out_hw(H, W)
        xpad = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p > 0 else x
        Wt = self._params["W"]
        out = np.zeros((B, Hout, Wout, self.out_chans), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                xs = xpad[:, i:i + s * Hout:s, j:j + s * Wout:s, :]
                out = out + xs @ Wt[i, j]
        if self.use_bias:
            out = out + self._params["b"]
        self._cache = (xpad, Hout, Wout, x.shape)
        return out

    def backward(self, dout):
        xpad, Hout, Wout, xshape = self._cache
        B, H, W, Cin = xshape
        k, s, p = self.k, self.stride, self.padding
        Wt = self._params["W"]
        dW = np.zeros_like(Wt)
        dxpad = np.zeros_like(xpad)
        if self.use_bias:
            self.set_grad("b", dout.sum(axis=(0, 1, 2)))
        dout2 = dout.reshape(-1, self.out_chans)
        for i in range(k):
            for j in range(k):
                xs = xpad[:, i:i + s * Hout:s, j:j + s * Wout:s, :]
                dW[i, j] = xs.reshape(-1, Cin).T @ dout2
                dxpad[:, i:i + s * Hout:s, j:j + s * Wout:s, :] += dout @ Wt[i, j].T
        self.set_grad("W", dW)
        dx = dxpad[:, p:p + H, p:p + W, :] if p > 0 else dxpad
        return dx


class DepthwiseConv2d(Module):
    """Per-channel (depthwise, no cross-channel mixing) 2D convolution, NHWC layout, stride 1,
    'same' padding (k // 2). Only `channels * k * k` weights -- negligible extra params/FLOPs.

    Used as the TinyViT-style "local mixing" unit inserted between attention and the MLP in a
    `TransformerBlock` (see that class's `local_conv` option): injects a cheap spatial-locality
    prior that plain (global) self-attention on flattened patch tokens otherwise lacks, which
    is the structural change targeting the attention-collapse finding in the earlier report.
    """

    def __init__(self, channels, kernel_size=3, rng=None, bias=True):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.channels = channels
        self.k = kernel_size
        self.padding = kernel_size // 2
        # small init: this sits on a residual branch, so start close to a no-op contribution
        self.add_param("W", init.normal(
            (kernel_size, kernel_size, channels), std=0.02, rng=rng))
        self.use_bias = bias
        if bias:
            self.add_param("b", init.zeros((channels,)))
        self._cache = None

    def forward(self, x, training=True):
        B, H, W, C = x.shape
        k, p = self.k, self.padding
        xpad = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p > 0 else x
        Wt = self._params["W"]
        out = np.zeros_like(x)
        for i in range(k):
            for j in range(k):
                out = out + Wt[i, j] * xpad[:, i:i + H, j:j + W, :]
        if self.use_bias:
            out = out + self._params["b"]
        self._cache = (xpad, x.shape)
        return out

    def backward(self, dout):
        xpad, xshape = self._cache
        B, H, W, C = xshape
        k, p = self.k, self.padding
        Wt = self._params["W"]
        dW = np.zeros_like(Wt)
        dxpad = np.zeros_like(xpad)
        if self.use_bias:
            self.set_grad("b", dout.sum(axis=(0, 1, 2)))
        for i in range(k):
            for j in range(k):
                xs = xpad[:, i:i + H, j:j + W, :]
                dW[i, j] = (xs * dout).sum(axis=(0, 1, 2))
                dxpad[:, i:i + H, j:j + W, :] += Wt[i, j] * dout
        self.set_grad("W", dW)
        dx = dxpad[:, p:p + H, p:p + W, :] if p > 0 else dxpad
        return dx


class ConvPatchStem(Module):
    """TinyViT-inspired convolutional alternative to `PatchEmbedding`: instead of one
    non-overlapping stride-`patch_size` linear projection, apply `log2(patch_size)` stride-2
    3x3 convolutions (with GELU between, none after the last) that progressively halve spatial
    resolution down to the same `img_size / patch_size` patch grid. Gives the network a local/
    translation-equivariant inductive bias before the transformer stack -- plain patchify has
    none -- which should matter more on this project's small (45k-image, 100-class) training
    set than it would at ImageNet scale. Output shape matches `PatchEmbedding`: (B, num_patches,
    embed_dim), same patch ordering (row-major grid), so it's a drop-in replacement selected via
    `VisionTransformer(conv_stem=True)` / config `"conv_stem": true`.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, rng=None):
        super().__init__()
        n_layers = round(_math.log2(patch_size))
        assert 2 ** n_layers == patch_size, "ConvPatchStem requires patch_size to be a power of 2"
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        hidden = max(embed_dim // 2, in_chans * 4)
        chans = [in_chans] + [hidden] * (n_layers - 1) + [embed_dim]
        self.convs = []
        self.acts = []
        for i in range(n_layers):
            conv = self.add_child(
                f"conv{i}", Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1, rng=rng))
            self.convs.append(conv)
            if i < n_layers - 1:
                self.acts.append(self.add_child(f"act{i}", GELU()))

    def forward(self, x, training=True):
        B = x.shape[0]
        for i, conv in enumerate(self.convs):
            x = conv.forward(x, training)
            if i < len(self.acts):
                x = self.acts[i].forward(x, training)
        return x.reshape(B, self.num_patches, -1)

    def backward(self, dout):
        B = dout.shape[0]
        dx = dout.reshape(B, self.grid, self.grid, -1)
        for i in reversed(range(len(self.convs))):
            if i < len(self.acts):
                dx = self.acts[i].backward(dx)
            dx = self.convs[i].backward(dx)
        return dx


class PatchEmbedding(Module):
    """Splits an image into non-overlapping patches, optionally tokenizes them via a
    per-channel 2D DCT (see nn/dct.py), and linearly projects the result to embed_dim.

    Input x: (B, H, W, C) with H == W == img_size. Patch extraction is a pure
    reshape/transpose (a bijective rearrangement of elements), so its backward is
    just the inverse reshape/transpose of the upstream gradient.

    tokenizer="raw" (default): behavior is byte-for-byte unchanged from before this
      option existed -- flattened raw pixel values go straight into `proj`.
    tokenizer="dct": each patch's raw pixels are replaced by their 2D DCT-II
      coefficients (per channel) before `proj`. If dct_keep is None, all P*P
      coefficients are kept (same patch_dim as raw -- an apples-to-apples comparison
      of tokenization *representation* alone). If dct_keep is an int, only the
      lowest-frequency `dct_keep` coefficients per channel are kept (JPEG-style
      truncation), shrinking `proj`'s input size and MACs. Ported from `token/` --
      see `nn/dct.py` for the (parameter-free, exact-inverse) transform itself.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, rng=None,
                 tokenizer="raw", dct_keep=None):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        assert tokenizer in ("raw", "dct"), f"unknown tokenizer '{tokenizer}'"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.tokenizer = tokenizer
        self.dct_keep = dct_keep
        self.use_trunc = tokenizer == "dct" and dct_keep is not None

        if tokenizer == "dct":
            self.add_child("dct", DCT2D(patch_size))
            if self.use_trunc:
                self.add_child("trunc", DCTTruncate(patch_size, dct_keep))
                self.patch_dim = dct_keep * in_chans
            else:
                self.patch_dim = patch_size * patch_size * in_chans
        else:
            self.patch_dim = patch_size * patch_size * in_chans

        self.add_child("proj", Linear(self.patch_dim, embed_dim, rng=rng))
        self._in_shape = None

    def _to_blocks(self, x):
        """(B, H, W, C) -> (B, N, P, P, C): unflattened per-patch spatial blocks, so an
        optional DCT can act on the (P, P) axes before the final flatten to patch_dim."""
        B = x.shape[0]
        P, G, C = self.patch_size, self.grid, self.in_chans
        x = x.reshape(B, G, P, G, P, C)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # B, G, G, P, P, C
        x = x.reshape(B, G * G, P, P, C)
        return x

    def forward(self, x, training=True):
        self._in_shape = x.shape
        blocks = self._to_blocks(x)  # B, N, P, P, C
        if self.tokenizer == "dct":
            blocks = self.dct.forward(blocks, training)
            if self.use_trunc:
                patches = self.trunc.forward(blocks, training)  # B, N, keep_k*C
            else:
                B, N, P, P2, C = blocks.shape
                patches = blocks.reshape(B, N, P * P2 * C)
        else:
            B, N, P, P2, C = blocks.shape
            patches = blocks.reshape(B, N, P * P2 * C)
        return self.proj.forward(patches, training)

    def backward(self, dout):
        dpatches = self.proj.backward(dout)
        B = self._in_shape[0]
        P, G, C = self.patch_size, self.grid, self.in_chans

        if self.tokenizer == "dct":
            if self.use_trunc:
                dblocks = self.trunc.backward(dpatches)  # B, N, P, P, C
            else:
                dblocks = dpatches.reshape(B, G * G, P, P, C)
            dblocks = self.dct.backward(dblocks)  # B, N, P, P, C
        else:
            dblocks = dpatches.reshape(B, G * G, P, P, C)

        dx = dblocks.reshape(B, G, G, P, P, C)
        dx = dx.transpose(0, 1, 3, 2, 4, 5)
        dx = dx.reshape(self._in_shape)
        return dx


class ClsPosEmbed(Module):
    """Prepends a learnable [CLS] token and adds learnable absolute positional embeddings."""

    def __init__(self, num_patches, embed_dim, rng=None):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.add_param("cls_token", init.normal((1, 1, embed_dim), std=0.02, rng=rng))
        self.add_param(
            "pos_embed", init.normal((1, num_patches + 1, embed_dim), std=0.02, rng=rng)
        )

    def forward(self, x, training=True):
        B, _, D = x.shape
        cls = np.broadcast_to(self._params["cls_token"], (B, 1, D))
        out = np.concatenate([cls, x], axis=1)
        out = out + self._params["pos_embed"]
        return out

    def backward(self, dout):
        self.set_grad("pos_embed", dout.sum(axis=0, keepdims=True))
        self.set_grad("cls_token", dout[:, :1, :].sum(axis=0, keepdims=True))
        return dout[:, 1:, :]
