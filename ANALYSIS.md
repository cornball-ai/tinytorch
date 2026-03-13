# tinytorch Performance Analysis: Whisper Transcription

Benchmark: whisper-tiny on 7.5s JFK audio clip, CPU, 20-core machine.

## Final Result

```
Backend         Warm Mean   RT Factor
tinytorch              0.24s     31.06x
PyTorch (Python)    0.22s     33.91x

tinytorch vs Python: 91% (warm start)
```

Both produce identical output: "As not what your country can do for you,
as what you can do for your country."

## Starting Point

tinytorch 2.9s mean, highly variable (1.3s to 4.3s). PyTorch 0.22s stable.

## What We Changed (in chronological order)

### 1. Fused C++ control-flow functions

Moved multi-op sequences from R into single `.Call()` boundaries:

| Function | What it replaces | File |
|----------|-----------------|------|
| `C_encoder_forward` | conv stem + all encoder layers + final LN | `src/transformer.cpp` |
| `C_decoder_forward_step` | embedding + all decoder layers + LN + logits + argmax | `src/transformer.cpp` |
| `C_prepare_cross_caches` | cross-attn K/V projection for all layers | `src/transformer.cpp` |
| `C_greedy_decode` | entire autoregressive decode loop | `src/transformer.cpp` |

Each has `at::NoGradGuard` to skip autograd graph construction.

R wrappers and readable R fallback paths preserved for all of these
(`encoder.R`, `decoder.R`, `transcribe.R`). Dispatch is based on whether
packed weights exist (`!is.null(self$.encoder_global_w)`).

### 2. Thread control API

Added `torch_set_num_threads()` / `torch_get_num_threads()` (and interop
variants) wrapping `at::set_num_threads()` / `at::get_num_threads()`.

### 3. whisper_pipeline()

Load model once, transcribe many times. Matches PyTorch benchmark structure.

### 4. Cached special token lookups

`whisper_special_tokens()` was calling `hfhub::hub_download()` on every
transcription to read `added_tokens.json`. Added an in-memory cache
(`new.env(parent = emptyenv())`).

## What Actually Mattered

Impact of each fix on warm transcription time:

| Fix | Before | After | Notes |
|-----|--------|-------|-------|
| `OMP_NUM_THREADS=4` | 2.9s (variable) | 0.55s (stable) | Thread oversubscription on 20 cores |
| Cache `load_added_tokens()` | 0.55s | 0.24s | 200ms file I/O per call, called 2x per transcription |
| Fused C++ control flow | - | - | Improved stability, not raw speed |
| `at::NoGradGuard` | - | - | Marginal |
| `C_greedy_decode` (full loop in C++) | - | - | R loop overhead was ~0.05ms/token vs 5ms kernel/token |

The two fixes that mattered were a threading default and a caching bug.
The C++ fusion work provides architectural value (cleaner code paths,
fewer moving parts) but did not drive the benchmark improvement.

## What We Measured

### Stage breakdown (OMP_NUM_THREADS=4, after all fixes)

```
audio_to_mel:   0.020s
encode:         0.110s   (C_encoder_forward)
cross_caches:   0.007s   (C_prepare_cross_caches)
greedy_decode:  0.110s   (C_greedy_decode, 23 tokens)
text_decode:    0.002s
---
Total:          ~0.25s
```

### R decode loop overhead

Compared `C_greedy_decode` (full loop in C++) vs R loop calling
`C_decoder_forward_step` per token:

```
C_greedy_decode: 0.112s (23 tokens, 4.9ms/tok)
R loop:          0.114s (23 tokens, 5.0ms/tok)
```

Difference: ~0.1ms/token. For 23 tokens, that's 2ms total. The per-token
R overhead (list access, tensor creation, S3 dispatch) is negligible
relative to kernel compute time at whisper-tiny scale.

This would matter more for longer sequences or smaller models where
kernel time per token is lower.

### Thread scaling

MKL defaults to all available cores. On this 20-core machine, that
causes thread oversubscription for whisper-tiny's small matrices
(384-dim hidden state). Both tinytorch and PyTorch are affected equally:

```
                20 threads          4 threads
tinytorch          0.4-3.3s (var)     0.11s (stable)
PyTorch         0.2-2.1s (var)     0.10s (stable)
```

`OMP_NUM_THREADS` must be set before process start. `Sys.setenv()` from R
is too late -- OpenMP initializes its thread pool when R's BLAS loads at
startup. `at::set_num_threads()` updates the libtorch-side count but
doesn't fix contention with R's already-initialized BLAS threads.

Working approaches:
- Shell: `OMP_NUM_THREADS=4 r -e '...'`
- `~/.bashrc`: `export OMP_NUM_THREADS=4`

Does NOT work:
- `Sys.setenv(OMP_NUM_THREADS = "4")` before `library()` -- too late
- `tinytorch::torch_set_num_threads(4)` -- affects libtorch only, not R's BLAS
- `~/.Renviron` -- littler doesn't load it

## What openai-whisper Actually Does

Researched the source (github.com/openai/whisper):

- **No `torch.compile`**, no Dynamo, no Inductor, no JIT
- **No custom CUDA kernels** (.cu files)
- Two Triton kernels exist but only for word-level timestamp post-processing (DTW + median filter), not inference
- Encoder/decoder are plain `nn.Module` forward passes
- Only optimizations: `F.scaled_dot_product_attention` (which we also use) and KV caching via forward hooks
- `@torch.no_grad()` on decode entry points

Their speed comes from Python being a lower-overhead language binding
around the same libtorch C++ kernels, not from any compilation or fusion.

## Corrected Mental Model

What was wrong in earlier analysis:

1. **"GC is the bottleneck"** -- No. R's mark-and-sweep GC was not the
   primary issue. The variability came from thread oversubscription, and
   the fixed overhead came from uncached file I/O. GC matters in theory
   for wrapper churn, but was not measurable here.

2. **"Python's refcounting makes it faster"** -- Partially true but
   overstated. Immediate object cleanup is about lifetime management, not
   throughput. Python's speed advantage is lower per-op dispatch overhead
   (C-API method lookup vs R's S3 dispatch + SEXP wrapping), but for
   whisper-tiny with ~5ms/token kernel time, R's per-op overhead is <1%
   of total time.

3. **"Fused kernels close the gap"** -- Our C++ functions are
   control-flow consolidation, not kernel fusion. They batch many libtorch
   ops into fewer `.Call()` crossings. This is architecturally cleaner but
   for this workload the crossing overhead was already small relative to
   compute.

What actually matters:

- **Profile before optimizing.** The 200ms `hub_download()` call was
  invisible until we timed individual stages.
- **Thread configuration dominates on many-core machines.** Both Python
  and R suffer equally from MKL oversubscription on small models.
- **Per-op dispatch overhead scales with op count / compute ratio.** For
  whisper-tiny (5ms/token, ~50 ops/token), R's ~0.005ms/op overhead is
  noise. For a model doing 0.1ms/token with 50 ops, it would be 25% of
  runtime.

## Architecture After Changes

```
R user code
    |
    v
whisper_pipeline()          -- load once
    |
    v
pipe$transcribe()           -- per audio file
    |
    +-- audio_to_mel()      -- R: av + FFT + mel filterbank
    |
    +-- model$encode()      -- dispatches to:
    |     fused:  C_encoder_forward()       [one .Call()]
    |     or R:   conv1 -> conv2 -> layers  [many .Call()s]
    |
    +-- greedy_decode()     -- dispatches to:
    |     fused:  C_prepare_cross_caches()  [one .Call()]
    |             + C_greedy_decode()       [one .Call(), full loop]
    |     or R:   model$decode() per token  [many .Call()s per token]
    |
    +-- tokenizer$decode()  -- R: integer vector -> text
```

Fused path: 3 boundary crossings total (mel + encode + decode).
R path: ~350+ crossings (preserved as readable reference implementation).

## Files Modified

### tinytorch

| File | Change |
|------|--------|
| `src/transformer.cpp` | Added `C_encoder_forward`, `C_prepare_cross_caches`, `C_greedy_decode`; `at::NoGradGuard` on all fused functions |
| `src/tensor.cpp` | Added `C_torch_{set,get}_num_threads`, `C_torch_{set,get}_num_interop_threads` |
| `R/nn_functions.R` | R wrappers: `encoder_forward`, `prepare_cross_caches`, `greedy_decode`, thread API |

### Whisper

| File | Change |
|------|--------|
| `R/encoder.R` | `pack_fused_weights()` packs global encoder weights; `forward()` dispatches to fused path |
| `R/decoder.R` | `pack_generate_weights()` packs decoder + cross-attn K/V weights |
| `R/model.R` | `prepare_cross_caches()` dispatches to fused path |
| `R/transcribe.R` | `whisper_pipeline()`, `greedy_decode_fused()` uses `C_greedy_decode` |
| `R/config.R` | `.whisper_cache` for `load_added_tokens()` and `whisper_special_tokens()` |
| `scripts/benchmark.R` | Uses pipeline, documents thread requirements |
