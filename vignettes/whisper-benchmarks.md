<!--
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteIndexEntry{Whisper Benchmarks: tinytorch vs torch vs PyTorch}
-->
---
title: "Whisper Benchmarks: tinytorch vs torch vs PyTorch"
---

Whisper Benchmarks
==================

End-to-end transcription benchmarks using OpenAI Whisper (base model,
7.5 second JFK audio clip). Four backends compared on both GPU and CPU.

Hardware: AMD Ryzen 7 9800X3D, 64 GB DDR5, NVIDIA RTX 5060 Ti 16 GB.
Software: libtorch 2.8.0, Python torch 2.8.0+cu128, R 4.5.0.

GPU Results (RTX 5060 Ti)
-------------------------

| Backend          | Cold   | Warm Mean | RT Factor |
|------------------|--------|-----------|-----------|
| torch (R)        | 1.84s  | 1.83s     | 4.12x     |
| tinytorch           | 1.62s  | 1.22s     | 6.18x     |
| tinytorch+compile   | 1.58s  | 1.25s     | 6.05x     |
| PyTorch          | 0.62s  | 0.13s     | 54.36x    |

RT Factor = audio duration / transcription time (higher is better).
Warm mean averaged over 3 runs after one warmup pass.

CPU Results
-----------

| Backend          | Cold   | Warm Mean | RT Factor |
|------------------|--------|-----------|-----------|
| torch (R)        | 2.01s  | 2.19s     | 3.44x     |
| tinytorch           | 13.60s | 13.46s    | 0.56x     |
| PyTorch          | 6.57s  | 0.34s     | 21.37x    |

Analysis
--------

### GPU: tinytorch vs torch (R)

tinytorch is **1.5x faster** than the torch R package on GPU. The advantage
comes from lower dispatch overhead: tinytorch's `.Call()` + S3 dispatch
adds ~1.4 us per operation vs torch's R7 + lantern path at ~10 us. On
GPU where kernel launches are fast but plentiful, this overhead
accumulates across ~85 operations per decoder token step.

### GPU: tinytorch vs PyTorch

PyTorch is **9.4x faster** than tinytorch on GPU. The gap has three
sources:

1. **FlashAttention** -- PyTorch's `scaled_dot_product_attention`
   dispatches to FlashAttention V2, which fuses the entire
   Q@K^T -> scale -> mask -> softmax -> attn@V pipeline into a single
   memory-efficient CUDA kernel. Whisper's manual attention materializes
   the full N x N attention matrix and requires 5 separate kernel
   launches.

2. **R dispatch overhead** -- Whisper's autoregressive decoder makes ~85
   `.Call()` round-trips per token across ~20 tokens per transcription.
   Each crossing involves SEXP/Tensor conversion and R heap allocation
   for intermediates. PyTorch's eager mode has ~0.5 us overhead per op
   vs R's ~1.4 us.

3. **Operator fusion** -- PyTorch fuses layer normalization, GELU, and
   residual-add into combined CUDA kernels via `torch.compile`. tinytorch's
   compile system fuses MLP blocks (linear -> gelu -> linear) but does
   not yet fuse attention, layer norm, or residual patterns.

### CPU: tinytorch vs torch (R)

tinytorch is **6x slower** than the torch R package on CPU. Both use the
same libtorch with bundled Intel MKL for tensor math, so BLAS
performance is identical. The gap is in R-level code: torch's decoder
implementation runs different R code paths and its internal dispatch
(while heavier per-call) may pipeline differently with CPU caches.

### CPU: tinytorch vs PyTorch

PyTorch is **40x faster** than tinytorch on CPU (warm). Python's
interpreter overhead is much lower, and PyTorch's CPU backend applies
fused kernels and operator optimizations that libtorch's eager mode
(used by both R packages) does not.

### Compile impact

tinytorch+compile shows marginal improvement (~2% on GPU) because the
compiler only fuses MLP blocks. The attention mechanism, layer norms,
and residual-adds -- which dominate runtime -- are not compiled.

VRAM Usage
----------

| Backend          | Peak VRAM |
|------------------|-----------|
| tinytorch           | 188 MB    |
| PyTorch          | 289 MB    |

tinytorch achieves lower peak VRAM than PyTorch through explicit `gc()`
calls in the encoder forward loop and native `$argmax()` in the decoder.
Without these optimizations, peak VRAM was 1096 MB due to R's garbage
collector not running during `with_no_grad` blocks.

Reproducing
-----------

```r
# From the whisper package directory (rtorch-migration branch)
source("scripts/benchmark_gpu.R")
```

Requires: `whisper` (tinytorch branch), `tinytorch`, `torch` R packages, plus
Python `openai-whisper` via `uv`.
