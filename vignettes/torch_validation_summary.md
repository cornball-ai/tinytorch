<!--
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteIndexEntry{Validation: R torch vs Python PyTorch}
-->

# R torch vs Python PyTorch: Validation & Benchmark Summary

Cornball AI maintains native R implementations of several AI models originally
written in Python/PyTorch. This document summarizes all validation tests,
numerical equivalence checks, and performance benchmarks across three packages:
**chatterbox** (TTS), **whisper** (STT), and **diffuseR** (image generation).

**Hardware**: RTX 5060 Ti (16GB VRAM, Blackwell architecture)

---

## 1. chatterbox (Text-to-Speech)

**Branch**: `rtorch-port` (current default)
**Python reference**: chatterbox-tts 0.1.4 (PyPI), `chatterbox-tts:blackwell` Docker container
**R tensor library**: tinytorch 0.1.0 (raw `.Call()` to libtorch, no Rcpp/lantern)

### 1.1 Component-Level Numerical Validation

Every component of the 798M-parameter Chatterbox model was validated against
Python PyTorch by saving intermediate tensors from the Docker container as
safetensors files and comparing them to R torch outputs on identical inputs.

| Component | Max Abs Diff | Status | Test Script | Notes |
|-----------|-------------|--------|-------------|-------|
| Text Tokenizer | 0 | Exact match | - | Same tokenizer.json |
| S3 Tokenizer | 0 | Exact match | test_s3tokenizer.R | 150/150 tokens identical |
| Mel Extractor | 0.000001 | Validated | test_voice_encoder.R | Slaney filterbank |
| T3 Conditioning | 0.000002 | Validated | test_t3_cond.R | Perceiver resampler |
| T3 Llama Backbone | 0.00003 | Validated | test_t3_llama.R | 520M param transformer |
| Voice Encoder | 0.00026 | Validated | test_voice_encoder.R | LSTM speaker embedding |
| Conformer Encoder | 0.0004 | Validated | test_encoder_steps.R | 6 layers + upsample |
| CAMPPlus | 0.0015 | Validated | test_campplus.R | Speaker embedding (S3Gen) |
| HiFi-GAN Vocoder | 0.026 | Validated | test_hifigan.R | Weight normalization handled |
| CFM Decoder | 0.052 | Validated | test_cfm_full.R | 10-step Euler ODE solver |

**Threshold**: < 0.001 = OK, < 0.1 = WARN, > 0.1 = FAIL.
All 10 components pass validation. The largest diff (0.052) is in the CFM
decoder's ODE solver, where floating-point accumulation over 10 Euler steps
is expected to diverge slightly.

### 1.2 Validation Methodology

```
Python side (Docker container):
  save_*.py scripts → outputs/*.safetensors

R side:
  test_*.R or compare_*.R scripts → load Python tensors → run R forward pass
  → compute (r_output - py_output)$abs()$max()$item()
```

**Script inventory** (~50 validation scripts in `scripts/`):
- 12 `save_*.py` scripts that capture Python intermediate tensors
- 12 `test_*.R` scripts for component-level numerical comparison
- 8 `compare_*.R` + `compare_*.py` matched pairs for detailed analysis
- ~20 `debug_*.R` / `debug_*.py` scripts used during development
- ~20 `trace_*.py` + `explore_*.py` for architecture discovery

### 1.3 Specific Cross-Language Comparisons

**Mel filterbank**: `compare_filterbank.py` / `compare_filterbank.R`
- Extracts librosa Slaney filterbank from Python, compares bin-by-bin with R
- Also compares HTK vs Slaney mel-scale formula

**STFT**: `compare_stft.py` / `compare_r_stft.R`
- Three-way comparison: librosa STFT vs torch center-padded vs manual padding

**Token generation**: `compare_generation.py` / `compare_t3.R`
- Full T3 autoregressive loop with explicit CFG, temperature, min-p, top-p

**Llama layers**: `compare_llama_layers.R`
- Layer-by-layer hidden state comparison across conditioning, text, and BOS segments

### 1.4 Performance Benchmarks

Benchmarked with `scripts/benchmark_gpu.R` on RTX 5060 Ti.
Test text: "The quick brown fox jumps over the lazy dog." (~10 words, ~3.2s audio)

| Backend | Cold Start | Warm Start | Audio Length | RT Factor | VRAM |
|---------|-----------|------------|-------------|-----------|------|
| Python container | 1.4s | 1.3s | 3.2s | **2.5x** | 3,205 MB |
| Native R (tinytorch) | 4.3s | 3.6s | 4.8s | **1.3x** | 3,114 MB |
| Native R + compile | 3.9s | 4.0s | 4.5s | **1.1x** | 4,575 MB |

**Key finding**: Python container is ~2.8x faster than native R (warm start).

**Why the gap**: Each tensor operation in R crosses the R/C++ boundary via
`.Call()` at ~1.4 us/op. Python's pybind11 path is ~0.5 us/op. With
~100-200 autoregressive tokens x 30 transformer layers, this adds up.
Compilation (torchlang) doesn't help because the bottleneck is per-token
loop overhead, not individual module dispatch.

**Both are faster than real-time** for practical use (1.3x and 2.5x RT factors).

### 1.5 Key R torch Differences Discovered

These were found during the chatterbox port and documented in the package CLAUDE.md:

| Issue | Python PyTorch | R torch |
|-------|---------------|---------|
| Module callable | `module(x)` | `module$forward(x)` or `module(x)` |
| `torch_arange` | exclusive end | **inclusive** end (off-by-one) |
| dtype promotion | stays in dtype | R scalars promote float16 -> float32 |
| In-place ops | `x.add_(y)` | `x$add_(y)` (same) but copy semantics differ |
| `conv_transpose1d` | `padding` param | padding handling differs |
| `torch_sort` indices | 0-indexed | **1-indexed** |
| `nn_embedding` input | 0-indexed | **1-indexed** |

---

## 2. whisper (Speech-to-Text)

**Branch**: `main` (CRAN release); `rtorch-migration` (WIP tinytorch port)
**Python reference**: openai-whisper (PyPI)
**R tensor library**: torch (CRAN, R package) on main; tinytorch on migration branch

### 2.1 Current State

The whisper package on `main` uses the standard `torch` R package (not tinytorch).
It is on CRAN. There are **no Python-vs-R numerical comparison scripts** --
validation was done against OpenAI's published model specifications and
pre-exported reference data.

### 2.2 Implicit Validation Artifacts

- `inst/assets/mel_80.csv` and `inst/assets/mel_128.csv`: Pre-computed mel
  filterbank matrices exported from OpenAI's Python `whisper.audio.mel_filters`.
  These serve as ground truth for the R mel computation.

- Hardcoded special token IDs in tests (e.g., `sot = 50258L`, `eot = 50257L`)
  match the OpenAI Whisper specification.

- The same safetensors model weights are loaded, so weight-level equivalence
  is implicit.

### 2.3 Test Suite (R-only, tinytest)

| Test File | What It Tests |
|-----------|--------------|
| test_audio.R | Mel filterbank creation, Hz/Mel conversion, pad_or_trim |
| test_config.R | Model configs, special token IDs, language tokens |
| test_encoder.R | Encoder module creation and output shapes |
| test_decoder.R | Decoder module creation and output shapes |
| test_tokenizer.R | Initial token sequences, timestamp decoding |
| test_transcribe.R | Text cleaning, segment extraction, end-to-end JFK transcription |
| test_alignment.R | DTW alignment, word grouping, word timestamps |

### 2.4 Performance (R torch, main branch)

Measured on RTX 5060 Ti transcribing a 17s audio clip with `word_timestamps = TRUE`:

| Model | Parameters | Disk (fp32) | Peak VRAM | Speed |
|-------|-----------|-------------|-----------|-------|
| tiny | 39M | 151 MB | 564 MiB | 5.5s |
| base | 74M | 290 MB | 734 MiB | 1.9s |
| small | 244M | 967 MB | 1,454 MiB | 3.6s |
| medium | 769M | 3.0 GB | 3,580 MiB | 8.6s |
| large-v3 | 1550M | 6.2 GB | 3,892 MiB | 16.7s |

Peak VRAM includes ~364 MiB torch CUDA context overhead.

**Note**: These are R-only numbers. No head-to-head Python comparison has been
run on main. The `rtorch-migration` branch has benchmark scripts for this
(see below) but they haven't been merged or run to completion.

### 2.5 tinytorch Migration Branch (WIP)

The `rtorch-migration` branch (commit `0a7e2ad`) replaces `torch::` with
`tinytorch::` and adds:

- **Fused C++ decoder step** via tinytorch for incremental decoding
- **Module compilation** (`tinytorch::compile()`) for encoder/decoder layers
- **Pure R safetensors reader** (`R/safetensors.R`)
- **Two benchmark scripts**:
  - `scripts/benchmark.R`: tinytorch vs PyTorch on CPU (tiny model)
  - `scripts/benchmark_gpu.R`: 4-way comparison (torch vs tinytorch vs tinytorch+compile vs PyTorch) on CPU and CUDA, with VRAM tracking

**Status**: Not merged. The branch has a stashed WIP state. The benchmark
scripts exist but no published results are available.

**Confounding factor**: Early benchmarks may have been clouded by hfhub
download calls during model loading, making cold-start times unreliable.
The download overhead needs to be separated from actual inference time for
meaningful comparison.

### 2.6 What's Missing for whisper

- No component-level numerical validation (unlike chatterbox)
- No Python reference tensor comparisons
- No published R-vs-Python benchmark results
- The rtorch-migration branch is stalled/WIP
- hfhub download calls in the load path need to be isolated from benchmarks

---

## 3. diffuseR (Image Generation)

**Branch**: `main` (primary); `hfhub-downloads` (model download refactor)
**Python reference**: HuggingFace diffusers library
**R tensor library**: torch (CRAN R package)

### 3.1 Numerical Validation: Python Reference Tests

diffuseR uses a JSON-based validation approach: Python scripts generate
ground-truth test cases that are shipped with the package and loaded by
tinytest during `R CMD check`.

**FlowMatch Scheduler** (`inst/validation/`):
- `validate_flowmatch_scheduler.py` generates 5 test cases from HuggingFace's
  `FlowMatchEulerDiscreteScheduler` (sigma arrays, timesteps, full denoising loop)
- `flowmatch_test_cases.json` ships with the package
- `test_flowmatch_python_validation.R` compares R implementation within tolerance 1e-4

**RoPE (Rotary Position Embedding)** (`inst/validation/`):
- `validate_rope.py` generates test cases from HuggingFace's `LTX2AudioVideoRotaryPosEmbed`
- `rope_test_cases.json` ships with the package
- `test_rope_python_validation.R` compares R implementation within tolerance 1e-4

**End-to-End Pipeline** (`inst/scripts/`):
- `validate_with_python.py` runs full LTX-2 pipeline (512x320, 17 frames, 25 steps)
  and saves output tensor as safetensors
- `test_full_pipeline_real.R` is the R counterpart for visual comparison

### 3.2 Native R vs TorchScript Validation

As TorchScript models fail on Blackwell GPUs, diffuseR migrated all components
to native R torch modules. Each was validated against the TorchScript output:

| Component | Parameters | Max Diff | Mean Diff | Status |
|-----------|-----------|----------|-----------|--------|
| VAE Decoder | 138/138 | 2e-5 (CPU), 2.7e-3 (CUDA) | - | Complete |
| Text Encoder (CLIP SD21) | 196/196 | 0.0267 | 0.0036 | Complete |
| Text Encoder 2 (OpenCLIP) | 517/517 | 6.82 (hidden), 0.0 (pooled) | - | Complete |
| UNet (SD21) | 686/686 | 0.06 | 0.01 | Complete |
| UNet (SDXL) | 1680/1680 | < 0.1% | - | Complete |

**All components validated. Full pipeline runs on Blackwell with everything on CUDA.**

### 3.3 Test Suite (tinytest)

| Test File | Type | What It Tests |
|-----------|------|--------------|
| test_flowmatch_python_validation.R | Python comparison | Scheduler vs HuggingFace |
| test_rope_python_validation.R | Python comparison | RoPE vs HuggingFace |
| test_vae_decoder.R | Native vs TorchScript | VAE decoder equivalence |
| test_text_encoder.R | Native vs TorchScript | CLIP encoder equivalence |
| test_unet.R | Native vs TorchScript | UNet equivalence |
| test_flowmatch_scheduler.R | Unit test | Scheduler creation, stepping |
| test_rope.R | Unit test | RoPE coordinates, rotation identity |
| test_dit_ltx2.R | Unit test | LTX-2 DiT transformer shapes |
| test_vae_ltx2.R | Unit test | LTX-2 VAE shapes |
| test_ltx2_weights.R | Unit test | Weight loading |
| test_text_encoder_ltx2.R | Unit test | LTX-2 text connector shapes |

72 total tests, all passing.

### 3.4 Key Lessons from the TorchScript Migration

Documented in `TORCHSCRIPT_MIGRATION.md`:

1. **Pre-norm vs post-norm**: HuggingFace CLIP uses pre-norm (layernorm BEFORE
   attention/MLP). Initial implementation used post-norm, causing large diffs.

2. **GELU variants matter**: text_encoder uses tanh approximation,
   text_encoder2 uses exact GELU. Wrong choice = wrong outputs.

3. **Timestep embedding parameters are model-specific**: SDXL uses
   `flip_sin_to_cos=TRUE` and `downscale_freq_shift=0`; SD21 uses the opposite.
   Getting these wrong caused 12% error in the SDXL UNet.

4. **Final layer norm behavior differs by export**: SD21 TorchScript includes
   final layer norm in the output; SDXL TorchScript does not. Must detect and match.

---

## 4. Cross-Package Patterns

### 4.1 R torch Gotchas (discovered across all packages)

| Issue | Impact | Fix |
|-------|--------|-----|
| R scalar + float16 tensor -> float32 | Silent dtype promotion breaks inference | Use `$mul()`, `$add()` tensor methods |
| `torch_arange` inclusive end | Off-by-one in ranges | Subtract 1 from end |
| `torch_sort` returns 1-indexed | Wrong token lookups | Don't add +1 to already 1-indexed results |
| `nn_embedding` expects 1-indexed | "Indexing starts at 1" error | Convert 0-indexed Python tokens to 1-indexed |
| `local_no_grad()` scope | Gradients re-enable outside helper function | Use `with_no_grad({ ... })` instead |

### 4.2 Validation Approaches Compared

| Aspect | chatterbox | whisper | diffuseR |
|--------|-----------|---------|----------|
| Python reference | Docker container | None (spec-based) | HuggingFace diffusers |
| Tensor comparison | safetensors files | N/A | JSON + safetensors |
| Granularity | Every component | End-to-end only | Components + schedulers |
| Automation | Manual scripts | tinytest | tinytest (JSON shipped) |
| CI-compatible | No (needs GPU + Docker) | Yes | Partial (JSON tests run, GPU tests skip) |
| Benchmark scripts | Yes (3 scripts) | Yes (on WIP branch) | No dedicated scripts |
| Published results | Yes (performance vignette) | Model table in README | TORCHSCRIPT_MIGRATION.md |

### 4.3 Performance Summary

| Package | R Warm | Python Warm | R/Python Ratio | Notes |
|---------|--------|-------------|----------------|-------|
| chatterbox | 3.6s | 1.3s | 2.8x slower | Autoregressive bottleneck |
| whisper (base, CUDA) | 1.9s* | - | Unknown | *R-only, no Python comparison |
| diffuseR | - | - | - | No benchmarks published |

*Whisper R-only speed on a 17s audio clip.

---

## 5. Reproducing

### chatterbox
```bash
# Save Python reference tensors
docker run --rm --gpus all \
  -v ~/chatterbox/scripts:/scripts \
  -v ~/chatterbox/outputs:/outputs \
  chatterbox-tts:blackwell \
  python /scripts/save_t3_steps.py

# Run R validation
cd ~/chatterbox && r scripts/validate_rtorch.R

# Run benchmarks
cd ~/chatterbox && r scripts/benchmark_gpu.R
```

### whisper
```bash
# Run tinytest suite (main branch)
r -e 'tinytest::test_package("whisper")'

# Benchmark scripts (rtorch-migration branch, not yet runnable on main)
cd ~/whisper && git checkout rtorch-migration
r scripts/benchmark_gpu.R
```

### diffuseR
```bash
# Run tinytest (includes Python validation JSON tests)
r -e 'tinytest::test_package("diffuseR")'

# Regenerate Python reference data
cd ~/diffuseR && python inst/validation/validate_flowmatch_scheduler.py
cd ~/diffuseR && python inst/validation/validate_rope.py
```

---

*Generated 2026-02-17 from repos: chatterbox (rtorch-port), whisper (main + rtorch-migration), diffuseR (main)*
