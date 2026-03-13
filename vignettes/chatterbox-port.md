<!--
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteIndexEntry{Porting Chatterbox TTS to tinytorch}
-->
---
title: "Porting Chatterbox TTS to tinytorch"
---

Porting Chatterbox TTS to tinytorch
=================================

This documents the work required to port
[Chatterbox TTS](https://github.com/resemble-ai/chatterbox) from the
`torch` R package to tinytorch. Chatterbox is a complete text-to-speech
engine with voice cloning, comprising ~5000 lines of R code across 12
files with 521 `tinytorch::` call sites.

The port required changes on both sides: new C++ ops and R modules in
tinytorch, and compatibility fixes in Chatterbox.

tinytorch: New C++ Operations
--------------------------

### Tensor Creation (src/tensor.cpp)

Six new functions wrapping ATen creation ops:

- `torch_arange` -- with inclusive end to match `torch` R package
- `torch_full`, `torch_linspace`
- `torch_ones_like`, `torch_zeros_like`, `torch_randn_like`
- `torch_tensor_from_buffer` -- create tensor from raw bytes (used by
  safetensors reader)

### Tensor Operations (src/ops.cpp)

Twelve new ops:

- `torch_cat`, `torch_clamp`, `torch_where`, `torch_sort`
- `torch_flip`, `torch_cumsum`, `torch_maximum`, `torch_multinomial`
- `torch_outer`, `torch_triu`, `torch_norm`, `torch_std`

### Complex & Signal Processing (src/ops.cpp)

Eight functions for audio processing:

- `torch_complex`, `torch_real`, `torch_imag`, `torch_polar`
- `torch_view_as_real`
- `torch_stft`, `torch_istft`, `torch_hann_window`

### NN Functional (src/nn_functions.cpp)

Nine new functions:

- `nnf_pad`, `nnf_interpolate`, `nnf_avg_pool1d`
- `nnf_softplus`, `nnf_normalize`
- `torch_conv_transpose1d`, `torch_conv2d`
- `torch_batch_norm`, `torch_lstm`

### Tensor Indexing (src/indexing.cpp)

Full `[.torch_tensor` and `[<-.torch_tensor` via ATen's
`tensor.index()` and `tensor.index_put_()`:

- R 1-based to C++ 0-based conversion
- Negative indices (Python convention, counting from end)
- Contiguous ranges detected and converted to `Slice`
- Non-contiguous vectors converted to index tensors
- Boolean mask indexing
- `drop = FALSE` support via unsqueeze

tinytorch: New R Modules
---------------------

### nn_module Infrastructure

- `$to(device, dtype)` -- no-op (CPU only), needed for weight loading
- `[[.nn_module` -- subscript access for `nn_module_list`
- `length.nn_module` -- module count

### NN Modules (R/nn.R)

- `nn_parameter`, `nn_buffer`
- `nn_module_list`, `nn_sequential`, `nn_identity`
- `nn_dropout`, `nn_sigmoid`, `nn_silu`, `nn_tanh`, `nn_elu`
- `nn_embedding`, `nn_conv1d`, `nn_conv_transpose1d`, `nn_conv2d`
- `nn_batch_norm1d`, `nn_batch_norm2d`
- `nn_lstm` -- multi-layer with bidirectional support
- `nn_upsample` -- wraps `nnf_interpolate`
- `with_autocast` -- no-op for CPU

### Tensor Methods (R/tensor.R)

New `$.torch_tensor` methods: `permute`, `expand`, `gather`,
`masked_fill`, `masked_fill_`, `copy_`, `norm`, `normal_`, `uniform_`,
`zero_`, `fill_`, `std`, `repeat_interleave`, `repeat`, `tolist`,
`index_select`, `narrow`, `scatter_`.

### Scalar-First Operators

All S3 operators (`+`, `-`, `*`, `/`, `^`, `%%`, `%/%`, `==`, `!=`,
`<`, `<=`, `>`, `>=`) now handle `scalar op tensor` in addition to
`tensor op scalar` and `tensor op tensor`.

Chatterbox: Compatibility Fixes
-------------------------------

### Dtype Constants (52 call sites)

The `torch` R package exposes dtype as zero-argument functions
(`torch_float32()`), while tinytorch uses constants (`torch_float32`).
All 52 occurrences across 12 files were updated:

```r
# Before (torch R package)
tinytorch::torch_tensor(x, dtype = tinytorch::torch_float32())

# After (tinytorch)
tinytorch::torch_tensor(x, dtype = tinytorch::torch_float32)
```

### LSTM Parameter Names (0-indexed)

tinytorch uses PyTorch's 0-indexed parameter names (`weight_ih_l0`),
matching the safetensors keys directly. The voice encoder weight
loader had a `+1` conversion that was wrong:

```r
# Before (assumed 1-indexed like old torch R package)
r_layer <- py_layer + 1
r_weight_ih <- paste0("weight_ih_l", r_layer)

# After (0-indexed, matching PyTorch and tinytorch)
r_weight_ih <- paste0("weight_ih_l", layer)
```

### Safetensors Reader

Chatterbox's `R/safetensors.R` was rewritten to parse the safetensors
binary format directly and create tinytorch tensors via
`torch_tensor_from_buffer`. This avoids the `safetensors` CRAN package
which creates `torch` package tensors (incompatible external pointer
format).

### Bang-Bang-Bang Splice

R's `!!!` operator (from rlang) doesn't work in base R:

```r
# Before
tinytorch::nn_sequential(!!!layers)

# After
do.call(tinytorch::nn_sequential, layers)
```

### Byte-Compiled Package Cache

After changing tinytorch, chatterbox must be reinstalled (`R CMD INSTALL`)
or the old byte-compiled code runs with stale references:

```bash
cd ~/tinytorch && R CMD INSTALL .
cd ~/chatterbox && R CMD INSTALL .  # Required!
```

Key Design Decisions
--------------------

### torch_arange: Inclusive End

The `torch` R package uses inclusive end semantics while ATen uses
exclusive. tinytorch matches the R package by computing the element count
and adjusting the exclusive end:

```cpp
int64_t n = (int64_t)std::floor((end - start) / step) + 1;
end = start + n * step;  // exclusive end for ATen
```

This correctly handles non-aligned endpoints:
`torch_arange(0, 511, 2)` produces 256 elements `[0, 2, ..., 510]`,
not 257.

### Tensor Indexing in C++

`[.torch_tensor` is implemented in C++ (`src/indexing.cpp`) rather than
pure R because:

1. ATen's `TensorIndex` / `Slice` API maps directly to R's index types
2. A pure R implementation would require many `.Call` round-trips
   (select, narrow, index_select per dimension)
3. Boolean mask handling and contiguous range detection are cleaner in
   C++

The R side uses `match.call(expand.dots = FALSE)` to detect missing
arguments (empty dimensions like `x[1,,]`), then passes a list of
index specs to C++.

### useDynLib in NAMESPACE

`tinyrox::document()` overwrites NAMESPACE but doesn't support
`@useDynLib`. The `useDynLib(tinytorch, .registration = TRUE)` line and
manual S3 registrations (`[`, `[<-`, `[[`) must be re-added after each
`document()` call. This is a known limitation to be addressed in tinyrox.

tinytorch: CUDA Device Support
----------------------------

The port required full CUDA support in tinytorch. All tensor creation
functions (`torch_zeros`, `torch_ones`, `torch_randn`, `torch_empty`,
`torch_full`, `torch_arange`, `torch_linspace`, `torch_hann_window`,
`torch_tensor`, `torch_tensor_from_buffer`) now accept a `device=`
parameter.

### Library Loading

CUDA libraries are conditionally linked. `Makevars.in` uses
`-Wl,--no-as-needed -ltorch_cuda -lc10_cuda` to force the libraries
into `DT_NEEDED`, since the linker would otherwise drop them (no
symbols are directly referenced from R code). The `configure` script
detects CUDA presence and sets this up automatically.

### nn_module$to(device, dtype)

Recursive device/dtype transfer for `nn_module`:

- Walks `private$parameters_`, `private$buffers_`, `private$modules_`
- Updates both the private lists and `self[[nm]]` references
- Preserves `nn_parameter` / `nn_buffer` class tags

### Scalar-First Operators on CUDA

S3 operators that handle `scalar op tensor` (e.g., `1.0 - x`) create
a temporary scalar tensor. These must be placed on the same device as
the tensor operand:

```r
# In Ops.torch_tensor
device <- .Call(C_tensor_device, e2)
temp <- torch_tensor(e1, dtype = ..., device = device)
```

### Memory Management

Two new functions expose libtorch's CUDA caching allocator:

- `cuda_mem_info()` -- free and total device memory (driver level)
- `cuda_memory_stats()` -- per-process allocated/reserved bytes
  (caching allocator level)
- `cuda_empty_cache()` -- release cached blocks back to the driver

These were essential for profiling Chatterbox's ~3 GB model on GPU.

Results
-------

With these changes, the full Chatterbox TTS pipeline (798M parameters)
runs on CUDA via tinytorch. The port validates component-by-component
against the Python reference implementation:

| Component             | Max Difference |
|-----------------------|---------------|
| Mel Spectrogram       | < 0.000001    |
| Voice Encoder         | < 0.000256    |
| S3 Tokenizer          | 100% match    |
| T3 Conditioning       | < 0.000002    |
| T3 Llama Backbone     | < 0.00003     |
| CAMPPlus Speaker Enc  | < 0.001471    |
| Conformer Encoder     | < 0.0004      |
| CFM Decoder           | < 0.028       |
| HiFi-GAN Vocoder      | < 0.026       |

Chatterbox is the largest application ported to tinytorch to date,
covering 521 `tinytorch::` call sites across ~5000 lines of R code.
