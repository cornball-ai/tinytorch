# tinytorch

High-performance R bindings to libtorch. Drop-in replacement for torch with minimal overhead.

## Architecture
- S3 class `torch_tensor` with `$` method dispatch via shared method table
- Rcpp for C++ / R glue (auto-registration, exception handling)
- No lantern, no R7
- Dual backend: real (libtorch) or stub (no-op) selected at configure time

## Build
```bash
r -e 'tinyrox::document(); tinypkgr::install()'
```

## Test
```bash
r -e 'tinytest::test_package("tinytorch")'
```

## Key conventions
- 1-based indexing for dimensions (converted to 0-based in C++)
- Negative dims pass through unchanged (PyTorch "from end" semantics)
