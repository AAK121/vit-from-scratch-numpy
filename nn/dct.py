"""Fixed (non-learnable) 2D DCT tokenizer components -- bonus "Image Tokenization Strategy".

Both classes here have NO learnable parameters (nothing is registered via add_param), so
they never show up in model.named_parameters() / the optimizer / the trainable-parameter
count printed by test.py. They are fixed linear transforms, exactly analogous in spirit to
PatchEmbedding's own reshape/transpose patch extraction (also parameter-free, also has a
trivial exact backward).

Forward:  Y = M @ X @ M^T   (2D DCT-II, orthonormal, applied per channel independently)
Backward: dX = M^T @ dY @ M

This is exact, not a finite-difference approximation: M is orthonormal (M @ M.T == I), so
M^T is *exactly* the inverse DCT. Because of that, both directions are literally the same
routine with M swapped for M^T -- see `_transform` below.
"""
import numpy as _np

from nn.backend import xp as np, to_device
from nn.module import Module


def _build_dct_matrix(P):
    """Orthonormal 1D DCT-II basis matrix, shape (P, P): y = M @ x is the orthonormal
    DCT-II transform of x (matches scipy.fft.dct(x, type=2, norm='ortho')). Row 0 is the
    constant/DC basis vector; higher rows are increasingly high-frequency cosines.
    """
    n = _np.arange(P)
    k = _np.arange(P).reshape(-1, 1)
    M = _np.cos(_np.pi / P * (n + 0.5) * k)
    M *= _np.sqrt(2.0 / P)
    M[0, :] *= 1.0 / _np.sqrt(2.0)
    return M.astype(_np.float32)


def zigzag_keep_mask(P, keep_k):
    """Boolean (P, P) mask selecting the `keep_k` lowest-frequency 2D-DCT coefficients
    (ordered by u+v, ties broken by u then v) -- a JPEG-style low-frequency truncation,
    applied identically to every channel.
    """
    assert 1 <= keep_k <= P * P
    coords = sorted((u + v, u, v) for u in range(P) for v in range(P))
    mask = _np.zeros((P, P), dtype=bool)
    for _, u, v in coords[:keep_k]:
        mask[u, v] = True
    return mask


class DCT2D(Module):
    """Per-channel 2D DCT-II on (..., P, P, C) patch blocks. Channel axis (last) is never
    mixed with the spatial (P, P) axes -- each channel's P x P block is transformed
    independently, matching how JPEG applies DCT per 8x8 block per channel plane.
    """

    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size
        M = _build_dct_matrix(patch_size)
        self._M = to_device(M)       # forward basis
        self._Mt = to_device(M.T)    # == inverse (orthonormal): M @ M.T == I

    @staticmethod
    def _transform(x, Mat):
        """Mat @ x @ Mat^T along axes (-3, -2); axis -1 (channel) passes through untouched."""
        t = np.einsum('ij,...jlc->...ilc', Mat, x)
        return np.einsum('...ilc,kl->...ikc', t, Mat)

    def forward(self, x, training=True):
        return self._transform(x, self._M)

    def backward(self, dout):
        return self._transform(dout, self._Mt)


class DCTTruncate(Module):
    """Selects the `keep_k` lowest-frequency DCT coefficients per channel (same fixed
    spatial positions for every channel), discarding the rest -- shrinks the tokenizer's
    output from P*P*C down to keep_k*C. This is a fixed 0/1 selection matrix, so backward
    is an exact scatter of gradient back to zeros at the dropped positions (their gradient
    really is exactly zero -- they never influenced the loss).
    """

    def __init__(self, patch_size, keep_k):
        super().__init__()
        self.P = patch_size
        self.keep_k = keep_k
        mask = zigzag_keep_mask(patch_size, keep_k)
        self._flat_idx = to_device(_np.flatnonzero(mask))  # (keep_k,) indices into P*P
        self._lead = None
        self._C = None

    def forward(self, x, training=True):
        # x: (..., P, P, C)
        lead, C = x.shape[:-3], x.shape[-1]
        flat = x.reshape(lead + (self.P * self.P, C))
        selected = flat[..., self._flat_idx, :]           # (..., keep_k, C)
        self._lead, self._C = lead, C
        return selected.reshape(lead + (self.keep_k * C,))

    def backward(self, dout):
        lead, C = self._lead, self._C
        dsel = dout.reshape(lead + (self.keep_k, C))
        dflat = np.zeros(lead + (self.P * self.P, C), dtype=dout.dtype)
        dflat[..., self._flat_idx, :] = dsel
        return dflat.reshape(lead + (self.P, self.P, C))
