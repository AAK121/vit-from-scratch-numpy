import math


def _conv_stem_macs(img_size, patch_size, in_chans, embed_dim):
    """MACs for `nn.layers.ConvPatchStem`: log2(patch_size) stride-2 3x3 convs, spatial
    resolution halving each layer, channels in_chans -> hidden -> ... -> embed_dim (see that
    class's channel schedule)."""
    n_layers = round(math.log2(patch_size))
    hidden = max(embed_dim // 2, in_chans * 4)
    chans = [in_chans] + [hidden] * (n_layers - 1) + [embed_dim]
    macs = 0
    hw = img_size
    for i in range(n_layers):
        hw_out = hw // 2
        macs += (hw_out * hw_out) * (3 * 3 * chans[i]) * chans[i + 1]
        hw = hw_out
    return macs


def count_flops_macs(config):
    img_size = config["img_size"]
    patch_size = config["patch_size"]
    in_chans = config["in_chans"]
    D = config["embed_dim"]
    L = config["depth"]
    num_classes = config["num_classes"]
    mlp_ratio = config.get("mlp_ratio", 4.0)
    conv_stem = config.get("conv_stem", False)
    local_conv = config.get("local_conv", False)
    tokenizer = config.get("tokenizer", "raw")
    dct_keep = config.get("dct_keep")

    P = (img_size // patch_size) ** 2  # num patches
    N = P + 1  # + CLS token
    patch_dim = patch_size * patch_size * in_chans
    if tokenizer == "dct" and dct_keep:
        patch_dim = dct_keep * in_chans
    mlp_hidden = int(D * mlp_ratio)

    macs = 0
    if conv_stem:
        macs_patch_embed = _conv_stem_macs(img_size, patch_size, in_chans, D)
    else:
        macs_patch_embed = P * patch_dim * D
    macs += macs_patch_embed

    # Tokenizer cost: raw pixels are "free" (a pure reshape). The DCT tokenizer applies a
    # separable 2D DCT-II per patch per channel -- two (patch_size x patch_size) matmuls,
    # i.e. 2 * patch_size^3 MACs per channel per patch. NOTE: truncation (dct_keep) shrinks
    # the *proj* input (patch_dim above), but the DCT itself is still computed in full before
    # low-frequency ones can be selected, so this cost is the same with or without dct_keep.
    dct_macs = 0
    if tokenizer == "dct" and not conv_stem:
        dct_macs = P * in_chans * 2 * (patch_size ** 3)
        macs += dct_macs

    # per transformer block: qkv proj + attention (QK^T, attn@V) + out proj + MLP(2 linears)
    macs_qkv = 3 * N * D * D
    macs_attn_scores = N * N * D  # Q K^T summed over heads
    macs_attn_out = N * N * D  # attn @ V summed over heads
    macs_out_proj = N * D * D
    macs_mlp = 2 * N * D * mlp_hidden
    macs_local_conv = P * 9 * D if local_conv else 0  # depthwise 3x3, spatial tokens only
    per_block = (macs_qkv + macs_attn_scores + macs_attn_out + macs_out_proj + macs_mlp
                 + macs_local_conv)
    macs += L * per_block

    macs_head = D * num_classes
    macs += macs_head

    flops = 2 * macs
    return dict(
        MACs=int(macs),
        FLOPs=int(flops),
        per_block_MACs=int(per_block),
        num_tokens=N,
        num_patches=P,
        patch_embed_MACs=int(macs_patch_embed),
        head_MACs=int(macs_head),
        dct_MACs=int(dct_macs),
        tokenizer=tokenizer,
    )
