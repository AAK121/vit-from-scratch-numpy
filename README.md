# Vision Transformer from Scratch in NumPy

A Vision Transformer for 100-class image classification, written in pure NumPy with hand-derived backpropagation for every layer. It reaches **55.30% validation accuracy** with **941,540 parameters**, and an optional CuPy backend runs the same code on a GPU.

![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy)
![No autograd](https://img.shields.io/badge/autograd-none-lightgrey)
![CuPy](https://img.shields.io/badge/GPU-CuPy%20(optional)-76B900?logo=nvidia)

## Overview

The task is 100-class classification on a 50,000-image subset of ImageNet (45,000 for training, 5,000 held out for validation), with a hidden test set of about 16,700 images that must be scored within one hour. PyTorch, TensorFlow, JAX and any other automatic-differentiation library are not allowed.

Three things make this hard:

- **No autograd.** Every forward pass needs a matching hand-derived backward pass: attention, LayerNorm, GELU, softmax cross-entropy with label smoothing, depthwise convolution and the DCT tokenizer. One wrong gradient silently caps accuracy.
- **Small data for a ViT.** 45,000 images for 100 classes (450 per class) is very little for a transformer, which lacks a CNN's built-in locality prior.
- **Compute.** NumPy on a CPU is slow. The model has to train in a practical time and still clear the 1-hour inference budget.

## Method

```
image 256×256×3
  └─ split into 16×16 patches (256 patches)
       └─ per-channel 2D DCT-II of each patch (fixed, parameter-free)
            └─ linear projection → 128-d tokens, + [CLS] token + learned positional embeddings  (257 tokens)
                 └─ 4 × pre-norm Transformer block
                 │     x  ← x + Dropout(MHSA(LN(x)))              4 heads
                 │     x  ← x + DWConv3×3(x[patches])             depthwise "local mixing" branch
                 │     x  ← x + Dropout(MLP(LN(x)))               GELU, ratio 4
                 └─ LN → [CLS] → linear head (100 classes)
```

**Model**
- ViT with embedding dim 128, depth 4, 4 heads and MLP ratio 4, giving 941,540 trainable parameters and 302.4M MACs (604.8M FLOPs) per image.
- **DCT tokenizer.** Each patch is replaced by its 2D DCT-II coefficients before projection, a fixed transform with an exact inverse (`nn/dct.py`).
- **Depthwise local convolution.** A 3×3 depthwise conv on the patch grid between attention and the MLP, in the style of TinyViT. It adds a cheap spatial-locality prior for only `128 × 9` parameters per block.

**Training recipe** (`configs/res256_v2.json`)
- AdamW with decoupled weight decay 0.05. Norm parameters, biases and the CLS and positional embeddings get no decay.
- 10-epoch linear warmup, then cosine decay of the learning rate (base 3e-4) over 150 epochs, with early stopping.
- Label smoothing 0.1, dropout 0.1 and Mixup (α = 0.2).
- 256 px input, with random 256 px crops taken from a 320 px training cache.

**Correctness and speed**
- `scripts/gradcheck.py` checks every layer's backward pass, the Mixup and label-smoothing losses, and the full model end to end against central finite differences. All 206 gradient checks pass.
- `nn/backend.py` swaps NumPy for CuPy when a GPU is present, with no change to the hand-written math. On the 64 px baseline, `test.py` ran **16.2× faster** on an NVIDIA RTX 4000 Ada than on the CPU, with identical accuracy.

## Results

Final checkpoint (`checkpoints/submission.{npz,json}`), evaluated on the 5,000-image validation split:

| Metric | Value |
| --- | --- |
| **Validation accuracy** | **55.30%** (2,765 / 5,000) |
| Trainable parameters | 941,540 |
| MACs / FLOPs per forward pass | 302,396,928 / 604,793,856 |
| Inference, GPU (5,000 images) | 74.9 s (66.8 img/s) |
| Inference, CPU (extrapolated) | ~877 s (19.0 img/s) |
| Peak GPU memory (batch 256) | 6.96 GB |
| CuPy GPU backend speed-up (64 px baseline) | 16.2× |

Both backends fit the 1-hour hidden-test budget: about 4.2 min on the GPU and 14.6 min on the CPU (extrapolated) for ~16,700 images.

### How accuracy got from 24.86% to 55.30%

| Configuration | Best val. acc. |
| --- | --- |
| 64 px, patch 8, random-crop augmentation (baseline) | 24.86% |
| 64 px, patch 8, no augmentation | 24.10% |
| 256 px, patch 16, random-crop augmentation | 28.12% |
| 256 px, patch 16, no augmentation | 29.22% |
| 256 px, patch 16, no augmentation + cosine LR | 33.42% |
| **256 px, patch 16, final recipe: AdamW + label smoothing + warmup/cosine + dropout + DCT tokenizer + local conv + Mixup** | **55.30%** |

The first five rows share the same 128-d, depth-4, 4-head network.

## Ablations and findings

What worked:
- **Higher input resolution.** At 64 px the pipeline shrank images about 6× from their median native size. Moving to 256 px with patch 16 (257 tokens rather than 65) gained 3 to 5 points at matched augmentation. Keeping patch 8 at 256 px would have meant 1,025 tokens and 33× the FLOPs of the 64 px model, so patch 16 was the practical choice.
- **A learning-rate schedule.** Cosine decay was the single largest gain in the first study: +4.2 points (29.22% → 33.42%). The constant-LR run plateaued and early-stopped at epoch 64, while the cosine run kept improving through all 150 epochs.
- **The v2 recipe.** AdamW, label smoothing, warmup, dropout, the DCT tokenizer, the depthwise local convolution and Mixup together took accuracy from 33.42% to 55.30%. The write-up does not break this gain down by component.
- **Scaling.** In 8-epoch sweeps, embedding width gave the cleanest accuracy-for-compute trade-off (64 → 128 → 256 dims: 19.4% → 21.1% → 23.9%). More heads at fixed width also helped (2 → 4 → 8 heads: 20.8% → 21.1% → 23.0%). Depth was not monotonic, and the 2-layer model scored best.

What failed:
- **Class-imbalanced sampling.** Training on a non-uniform subset raised training accuracy but roughly halved validation accuracy. At 50% of the data: train 30.0% vs 20.8%, val 10.2% vs 21.1%, compared with uniform sampling.
- **Random-crop augmentation at 256 px.** It helped at 64 px (+0.76 points) but hurt at 256 px (28.12% vs 29.22%). At 64 px a center crop often cut off the subject. At 256 px the subject is rarely lost, so the crop jitter only added noise.
- **Attention collapse.** Grad-CAM and Attention Rollout showed that the later blocks of the 64 px baseline collapse their attention. Gradient-based saliency therefore had to backpropagate through the full network to be meaningful.

Profiling: on a CPU at 64 px the elementwise GELU cost more time than all the matmuls combined (34.6% vs 32.2% of runtime). At 256 px, attention takes over as the largest cost (34.2%), as its O(N²) scaling predicts.

## Repository structure

```
.
├── nn/                     # the framework: every layer has forward() and a hand-derived backward()
│   ├── backend.py          #   NumPy / CuPy switch (VIT_DEVICE=cpu|auto)
│   ├── module.py           #   parameter registry, train/eval mode
│   ├── layers.py           #   Linear, LayerNorm, GELU, Dropout, PatchEmbedding, DepthwiseConv2d, ...
│   ├── attention.py        #   multi-head self-attention
│   ├── dct.py              #   2D DCT-II patch tokenizer
│   ├── transformer.py      #   TransformerBlock, VisionTransformer, build_vit()
│   ├── losses.py           #   softmax cross-entropy with label smoothing
│   └── optim.py            #   Adam / AdamW
├── data/dataset.py         # image cache, loaders, random crop, Mixup
├── utils/                  # checkpoint I/O, MAC/FLOP counter, metrics, CSV logging
├── scripts/
│   ├── gradcheck.py        # finite-difference check of every backward pass
│   ├── smoke_test.py       # overfit a handful of images end to end
│   ├── build_cache.py      # one-time resize/cache of the dataset
│   ├── run_model_scaling.py, run_data_scaling.py
│   ├── profiling.py        # cProfile + per-component timing
│   ├── visualize.py        # Grad-CAM and Attention Rollout
│   └── make_plots*.py, data_distribution.py
├── configs/                # base.json (64 px), res256.json, res256_v2.json (final)
├── checkpoints/submission.{npz,json}   # final 55.30% model (3.6 MB)
├── train.py
├── test.py
├── setup.bash
└── requirements.txt
```

## How to run

**Setup.** This installs `requirements.txt` into the active venv or conda environment, or into a new `.venv`. The pinned CuPy build targets CUDA 13; on a CPU-only machine `numpy`, `pillow`, `tqdm` and `scipy` are enough.

```bash
bash setup.bash
```

**Evaluate the released checkpoint.** The dataset follows the ImageFolder layout, with one sub-folder per class (WordNet IDs).

```bash
python test.py --test-path /absolute/path/to/dataset
# prints accuracy, inference time, throughput, MACs, FLOPs and parameter count
VIT_DEVICE=cpu python test.py --test-path /absolute/path/to/dataset   # force the NumPy backend
```

**Verify the gradients.**

```bash
python scripts/gradcheck.py
```

**Train the final model from scratch.** Put the training images in `train/` (ImageFolder layout), or change `data_root` in the config.

```bash
python scripts/build_cache.py --config configs/res256_v2.json      # 256 px eval + 320 px train cache
python train.py --config configs/res256_v2.json --tag res256_v2_dct_local_mixup \
    --epochs 150 --early-stop-patience 15 --lr-schedule cosine
cp checkpoints/res256_v2_dct_local_mixup/best.npz  checkpoints/submission.npz
cp checkpoints/res256_v2_dct_local_mixup/best.json checkpoints/submission.json
```

The 64 px baseline and the scaling studies use `configs/base.json`, `scripts/run_model_scaling.py` and `scripts/run_data_scaling.py`.

## Team and acknowledgements

- **Author:** Alok Kale
- Built for **GNR650: Advanced Topics in Deep Learning for Image Analysis**, Indian Institute of Technology Bombay, under the guidance of **Prof. Biplab Banerjee**.
- Claude (Anthropic) was used as an AI pair-programming assistant during implementation and debugging.

## References

1. A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
2. A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
3. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019.
4. I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
5. H. Zhang, M. Cisse, Y. Dauphin, D. Lopez-Paz. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018.
6. C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision* (label smoothing). CVPR 2016.
7. K. Wu et al. *TinyViT: Fast Pretraining Distillation for Small Vision Transformers.* ECCV 2022.
8. S. Abnar, W. Zuidema. *Quantifying Attention Flow in Transformers* (Attention Rollout). ACL 2020.
9. R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
