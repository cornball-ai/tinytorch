<!--
%\VignetteEngine{simplermarkdown::mdweave_to_html}
%\VignetteIndexEntry{Benchmarks: tinytorch vs torch vs Python}
-->
---
title: "Benchmarks: tinytorch vs torch vs Python"
---

Benchmarks
----------

All benchmarks use libtorch 2.8.0 on CPU. Times are microseconds per call,
averaged over 10,000 iterations (200 for large matmul). Lower is better.

Four implementations compared:

- **Python** -- `torch 2.8.0+cpu` via pip (CPython 3.10)
- **R torch** -- `torch 0.16.3.9000` (Rcpp + lantern + R7 dispatch)
- **Rcpp** -- minimal `Rcpp::sourceCpp` wrapper, direct to libtorch (no lantern, no R7)
- **tinytorch** -- raw R C API + S3, direct to libtorch (no Rcpp, no lantern, no R7)

### Results

| Test                         | Python | R torch | Rcpp  | tinytorch |
|------------------------------|--------|---------|-------|-----------|
| Function add (10x10)         | 0.5    | 10.1    | 1.5   | 1.3       |
| Method .add (10x10)          | 0.5    | 13.3    | --    | 3.2       |
| Chained matmul (10x10)       | 1.3    | 19.1    | 3.3   | 3.4       |
| Method chain .add().mul()    | 0.9    | 24.9    | --    | 5.3       |
| Creation randn(10,10)        | 0.8    | 16.9    | 1.7   | 1.7       |
| Large matmul (1000x1000)     | 1,371  | 175,545 | 175,380 | 176,005 |

All times in microseconds. Rcpp has no `$` dispatch so method tests are omitted.

### What the numbers mean

**Overhead-dominated operations** (small tensors): tinytorch is 7-10x faster than
R torch and within 3x of Python. The ~1 us floor is R's `.Call()` overhead.
Rcpp and tinytorch are identical here -- Rcpp adds no measurable cost vs raw C.

**Compute-dominated operations** (1000x1000 matmul): all R implementations converge
since the time is spent inside libtorch BLAS. The gap vs Python (1.4 ms vs 176 ms)
is a BLAS difference -- Python's pip wheel ships Intel MKL, while R torch's bundled
libtorch uses a slower default. This is not an R overhead issue.

### Where the overhead lives

```
Python:   R function --> pybind11 --> libtorch          ~0.5 us
tinytorch: R function --> .Call() --> libtorch           ~1.4 us
Rcpp:     R function --> .Call() --> Rcpp glue --> libtorch  ~1.5 us
R torch:  R function --> R7 $ --> call_c_function --> Rcpp --> lantern --> libtorch  ~10 us
```

The R torch overhead comes from R7 class dispatch, dynamic function name resolution
(`call_c_function`), and the lantern shim layer (dlopen'd function pointers). Removing
those layers gets R within 3x of Python.

### Reproducing

```r
# Run from the tinytorch package directory
source("bench/benchmark.R")
# Results saved to bench/results.csv
```

Requires: `torch`, `tinytorch`, `Rcpp`, `jsonlite` R packages, plus Python torch
2.8.0 in a venv at `/tmp/torch-bench/`.
