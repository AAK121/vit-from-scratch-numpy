import numpy as np  

from nn.backend import xp
from nn.module import Module
from nn.layers import (
    Linear, LayerNorm, GELU, Dropout, PatchEmbedding, ClsPosEmbed, ConvPatchStem,
    DepthwiseConv2d,
)
from nn.attention import MultiHeadSelfAttention


class TransformerBlock(Module):
    """Pre-norm block: x1 = x + Drop(Attn(LN(x))) ; out = x1 + Drop(MLP(LN(x1))).

    `dropout` (default 0.0, a no-op -- see `nn.layers.Dropout`) applies to each sub-layer's
    output before it joins the residual stream, matching the standard ViT/DeiT placement
    (`proj_drop` after attention, `drop` after the MLP's second linear).

    `local_conv_grid` (default None, a no-op): when set to the spatial grid side length (e.g.
    8 for an 8x8 patch grid), inserts a TinyViT-style 3x3 depthwise-conv "local mixing" unit as
    a second residual branch on x1 -- `x1 <- x1 + DWConv(x1[patches])` (the CLS token, which
    has no spatial position, passes through unchanged) -- right after the attention residual
    and before LN2/MLP, matching TinyViT's placement "between attention and MLP". This costs
    only `embed_dim * 9` extra params/block and gives the (otherwise purely global) attention
    stack a cheap spatial-locality prior.
    """

    def __init__(self, embed_dim, num_heads, mlp_hidden, rng=None, dropout=0.0,
                 local_conv_grid=None):
        super().__init__()
        self.add_child("ln1", LayerNorm(embed_dim))
        self.add_child("attn", MultiHeadSelfAttention(embed_dim, num_heads, rng=rng))
        self.add_child("drop1", Dropout(dropout))
        self.local_conv_grid = local_conv_grid
        if local_conv_grid is not None:
            self.add_child("local_conv", DepthwiseConv2d(embed_dim, kernel_size=3, rng=rng))
        self.add_child("ln2", LayerNorm(embed_dim))
        self.add_child("fc1", Linear(embed_dim, mlp_hidden, rng=rng))
        self.add_child("act", GELU())
        self.add_child("fc2", Linear(mlp_hidden, embed_dim, rng=rng))
        self.add_child("drop2", Dropout(dropout))

    def _local_conv_fwd(self, x1, training):
        B, N, D = x1.shape
        G = self.local_conv_grid
        cls_tok = x1[:, :1, :]
        spatial = x1[:, 1:, :]
        grid = spatial.reshape(B, G, G, D)
        conv_out = self.local_conv.forward(grid, training).reshape(B, N - 1, D)
        return xp.concatenate([cls_tok, spatial + conv_out], axis=1)

    def _local_conv_bwd(self, dx1_mod):
        B, N, D = dx1_mod.shape
        G = self.local_conv_grid
        dcls = dx1_mod[:, :1, :]
        dspatial_out = dx1_mod[:, 1:, :]  # grad wrt (spatial + conv(spatial))
        dgrid = self.local_conv.backward(dspatial_out.reshape(B, G, G, D))
        dspatial = dspatial_out + dgrid.reshape(B, N - 1, D)
        return xp.concatenate([dcls, dspatial], axis=1)

    def forward(self, x, training=True):
        h = self.ln1.forward(x, training)
        h = self.attn.forward(h, training)
        h = self.drop1.forward(h, training)
        x1 = x + h
        if self.local_conv_grid is not None:
            x1 = self._local_conv_fwd(x1, training)

        h2 = self.ln2.forward(x1, training)
        h2 = self.fc1.forward(h2, training)
        h2 = self.act.forward(h2, training)
        h2 = self.fc2.forward(h2, training)
        h2 = self.drop2.forward(h2, training)
        out = x1 + h2
        return out

    def backward(self, dout):
        # out = x1 + h2
        dx1 = dout.copy()
        dpath2 = self.drop2.backward(dout)
        dpath2 = self.fc2.backward(dpath2)
        dpath2 = self.act.backward(dpath2)
        dpath2 = self.fc1.backward(dpath2)
        dpath2 = self.ln2.backward(dpath2)
        dx1 = dx1 + dpath2  # total grad wrt (post-local-conv) x1

        if self.local_conv_grid is not None:
            dx1 = self._local_conv_bwd(dx1)  # grad wrt pre-local-conv x1 (i.e. x + h)

        # x1 = x + h
        dx0 = dx1.copy()
        dpath1 = self.drop1.backward(dx1)
        dpath1 = self.attn.backward(dpath1)
        dpath1 = self.ln1.backward(dpath1)
        dx0 = dx0 + dpath1
        return dx0


class VisionTransformer(Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        seed=0,
        dropout=0.0,
        conv_stem=False,
        local_conv=False,
        tokenizer="raw",
        dct_keep=None,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)
        if conv_stem:
            assert tokenizer == "raw", "tokenizer (DCT) applies to PatchEmbedding only; " \
                "ConvPatchStem (conv_stem=True) operates directly on raw pixels"
            self.add_child(
                "patch_embed", ConvPatchStem(img_size, patch_size, in_chans, embed_dim, rng=rng)
            )
        else:
            self.add_child(
                "patch_embed", PatchEmbedding(img_size, patch_size, in_chans, embed_dim, rng=rng,
                                               tokenizer=tokenizer, dct_keep=dct_keep)
            )
        self.add_child(
            "cls_pos", ClsPosEmbed(self.patch_embed.num_patches, embed_dim, rng=rng)
        )
        self.add_child("pos_drop", Dropout(dropout))
        grid = img_size // patch_size
        self.blocks = []
        for i in range(depth):
            # local_conv only modifies the *spatial* tokens (CLS passes through untouched, see
            # TransformerBlock), and LayerNorm/Linear never mix across tokens -- so a spatial
            # change only reaches the CLS-only readout used by the loss if some *later* block's
            # attention subsequently lets CLS attend to it. On the last block there is no later
            # attention, so local_conv there would be a dead parameter (verified: its gradient
            # is exactly zero in gradcheck). Skip it on the final block.
            blk_local_conv_grid = grid if (local_conv and i < depth - 1) else None
            blk = TransformerBlock(embed_dim, num_heads, int(embed_dim * mlp_ratio), rng=rng,
                                    dropout=dropout, local_conv_grid=blk_local_conv_grid)
            self.add_child(f"block{i}", blk)
            self.blocks.append(blk)
        self.add_child("ln_final", LayerNorm(embed_dim))
        self.add_child("head", Linear(embed_dim, num_classes, rng=rng))

        self.config = dict(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            seed=seed,
            dropout=dropout,
            conv_stem=conv_stem,
            local_conv=local_conv,
            tokenizer=tokenizer,
            dct_keep=dct_keep,
        )
        self._cls_shape = None

    def forward(self, x, training=True):
        x = self.patch_embed.forward(x, training)
        x = self.cls_pos.forward(x, training)
        x = self.pos_drop.forward(x, training)
        for blk in self.blocks:
            x = blk.forward(x, training)
        x = self.ln_final.forward(x, training)
        self._cls_shape = x.shape
        cls = x[:, 0, :]
        logits = self.head.forward(cls, training)
        return logits

    def backward(self, dlogits):
        dcls = self.head.backward(dlogits)  # (B, D)
        dx = xp.zeros(self._cls_shape, dtype=dcls.dtype)
        dx[:, 0, :] = dcls
        dx = self.ln_final.backward(dx)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        dx = self.pos_drop.backward(dx)
        dx = self.cls_pos.backward(dx)
        dx = self.patch_embed.backward(dx)
        return dx


def build_vit(config):
    keys = ("img_size", "patch_size", "in_chans", "num_classes", "embed_dim",
            "depth", "num_heads", "mlp_ratio", "seed", "dropout", "conv_stem", "local_conv",
            "tokenizer", "dct_keep")
    kwargs = {k: config[k] for k in keys if k in config}
    return VisionTransformer(**kwargs)
