# tinytorch 0.2.0

## Preparing for CRAN

* First submission target. Package now builds cleanly against 'libtorch' 2.11.0 on Linux and falls back to a stub backend on other Unix platforms so install always succeeds.
* `DESCRIPTION` rewritten for CRAN: single-quoted software names, angle-bracketed upstream URL, ORCID, and `Cornball AI` as copyright holder matching `LICENSE`.
* Removed the `glue` import. The package now has a single hard dependency on `Rcpp`.
* Dropped `R/zzz-compat-ops.R`, a stale set of 38 auto-generated wrappers copied from the 'torch' R package that were never wired up and relied on helpers tinytorch does not provide.
* Fixed a parse warning in `torch_iinfo()` caused by the `-2147483648L` integer literal.
* `cuda_synchronize()`, `torch_manual_seed()`, and `torch_scaled_mm()` now actually call libtorch. Previous versions had the C++ implementations in `src/gen-ops.cpp` but never registered them with R, so the R wrappers were effectively stubs. `torch_manual_seed()` now seeds libtorch's global generator (confirmed reproducible across `torch_randn()` calls). `cuda_synchronize()` walks every CUDA device and calls `cudaDeviceSynchronize()` — necessary for correct benchmarking and for surfacing async CUDA errors at a specific line.

## Dispatch and coverage

* Full dispatch table coverage plus autograd and optimizers (#5).
* Added FP8 dtypes and coverage tests (#8).
* Bumped bundled libtorch target to 2.11.0 (#7).
* Exported 38 missing `torch_*` namespace functions (#9).

# tinytorch 0.1.0

* Initial development release. Direct R bindings to 'libtorch' with a
  'torch'-compatible API.
