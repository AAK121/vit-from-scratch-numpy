"""Multi-head self-attention, forward and hand-derived backward. No autodiff.

Forward (per head): scores = (Q K^T) * scale ; attn = softmax(scores) ; out = attn V
Backward derivations used below (standard, verified by scripts/gradcheck.py):
  out = attn @ V            => dV = attn^T @ dout   ,  dattn = dout @ V^T
  attn = softmax(scores)    => dscores = attn * (dattn - sum(dattn*attn, axis=-1))
  scores = (Q K^T) * scale  => draw = dscores * scale
                                dQ = draw @ K ,  dK = draw^T @ Q
"""
import math as _math

from nn.backend import xp as np
from nn.module import Module
from nn.layers import Linear


class MultiHeadSelfAttention(Module):
    def __init__(self, embed_dim, num_heads, rng=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / _math.sqrt(self.head_dim)
        self.add_child("qkv", Linear(embed_dim, 3 * embed_dim, rng=rng))
        self.add_child("proj", Linear(embed_dim, embed_dim, rng=rng))
        self._cache = None

    def _split_heads(self, x):
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim
        return x.reshape(B, N, H, d).transpose(0, 2, 1, 3)  # B,H,N,d

    def _merge_heads(self, x):
        B, H, N, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, N, H * d)

    def forward(self, x, training=True):
        qkv = self.qkv.forward(x, training)  # B,N,3D
        q, k, v = np.split(qkv, 3, axis=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale  # B,H,N,N
        scores = scores - scores.max(axis=-1, keepdims=True)
        expv = np.exp(scores)
        attn = expv / expv.sum(axis=-1, keepdims=True)

        out = np.matmul(attn, v)  # B,H,N,d
        merged = self._merge_heads(out)
        result = self.proj.forward(merged, training)

        self._cache = (q, k, v, attn)
        return result

    def backward(self, dout):
        q, k, v, attn = self._cache
        dmerged = self.proj.backward(dout)  # B,N,D
        dout_heads = self._split_heads(dmerged)  # B,H,N,d

        dattn = np.matmul(dout_heads, v.transpose(0, 1, 3, 2))  # B,H,N,N
        dv = np.matmul(attn.transpose(0, 1, 3, 2), dout_heads)  # B,H,N,d

        draw = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        draw = draw * self.scale

        dq = np.matmul(draw, k)  # B,H,N,d
        dk = np.matmul(draw.transpose(0, 1, 3, 2), q)  # B,H,N,d

        dq = self._merge_heads(dq)
        dk = self._merge_heads(dk)
        dv = self._merge_heads(dv)
        dqkv = np.concatenate([dq, dk, dv], axis=-1)  # B,N,3D
        dx = self.qkv.backward(dqkv)
        return dx
