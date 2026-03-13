# tinytorch

Linux-only R bindings to [libtorch](https://pytorch.org/cplusplus/) (the PyTorch C++ distribution). Provides a `torch`-compatible API with minimal R-side dependencies (just Rcpp). Much of the simplicity comes from targeting a single platform: no cross-platform build machinery, no download-at-install-time dance, just a configure script that finds libtorch and links against it.

## Installation

```r
~~install.packages("tinytorch")~~
# CRAN publication forthcoming

# Development version
remotes::install_github("cornball-ai/tinytorch")
```

On non-Linux systems, tinytorch builds in stub mode: the package loads but `is_available()` returns `FALSE`.

### Installing libtorch

```r
tinytorch::install_libtorch()
# Then reinstall tinytorch to pick up the backend
install.packages("tinytorch")
```

This downloads the CPU build to `~/.local/lib/libtorch`. For CUDA, install libtorch manually and point `LIBTORCH_HOME` at it.

The configure script searches for libtorch in order:

1. `$LIBTORCH_HOME` env var
2. `~/.local/lib/libtorch`
3. The `torch` R package's bundled copy

### System requirements

libtorch requires glibc >= 2.29. In practice:

| Distro | Minimum version |
|--------|----------------|
| Ubuntu | 20.04 |
| Debian | 11 |
| RHEL / CentOS | 9 |
| Fedora | 31 |

Older systems will fail at link time or with symbol errors at runtime.

## Usage

```r
library(tinytorch)

is_available()
#> TRUE

# Create tensors
x <- torch_randn(3, 4)
y <- torch_ones(3, 4)

# Operations
z <- x + y
torch_mm(x$t(), y)

# Neural network modules
model <- nn_sequential(
    nn_linear(784, 128),
    nn_relu(),
    nn_linear(128, 10)
)
```

## Expression compiler

tinytorch includes a compiler that fuses R torch expressions into optimized SIMD kernels:

```r
f <- compile(function(x, y) (x + y) * torch_exp(-x))
f(torch_randn(1000), torch_randn(1000))
```

## License

MIT
