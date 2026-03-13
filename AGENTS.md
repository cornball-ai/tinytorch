# Repository Guidelines

## Project Structure & Module Organization
This is an R package with C++/libtorch bindings.
- `R/`: R API, S3 classes (`torch_tensor`) and compile/trace utilities.
- `src/`: C++/Rcpp bridge and kernels (`C_torch_*` wrappers, fused ops).
- `inst/tinytest/`: tinytest test suite (`test_*.R`).
- `man/`: generated Rd docs (via roxygen/tinyrox).
- `vignettes/`: longer-form docs and benchmarks.
- `bench/`: performance benchmarks and scripts.

## Build, Test, and Development Commands
- `r -e 'tinyrox::document(); tinypkgr::install()'`: regenerate docs and install the package locally.
- `r -e 'tinytest::test_package("tinytorch")'`: run the tinytest suite.
- Optional: `torch_lint(quote({ ... }))` (see `?torch_lint`) to flag graph breaks in compiled expressions.

## Coding Style & Naming Conventions
- R: 2-space indentation, snake_case functions, roxygen headers for exports.
- C++: follow existing formatting in `src/` (indentation and brace style), keep wrappers named `C_torch_*`.
- Favor explicit conversions between 1-based R dims and 0-based libtorch dims (see `src/` patterns).
- Keep API-compatible behavior with the `torch` package.

## Testing Guidelines
- Framework: tinytest (`inst/tinytest/`).
- Naming: `test_*.R` files with focused, deterministic cases.
- Add tests for new ops, tracing/compile paths, and graph optimizations.

## Commit & Pull Request Guidelines
- Commits are short, imperative, and descriptive (e.g., “Add GPU kernel launch”).
- PRs should include: a concise summary, test command(s) run, and any GPU/CUDA requirements or benchmark notes.
- Link related issues when applicable.

## Dependencies & Configuration
- Requires R (>= 4.1.0) and libtorch from an installed `torch` package.
- See `CLAUDE.md` for architecture notes and the canonical build/test commands.
