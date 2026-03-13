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

See the [PyTorch install page](https://pytorch.org/get-started/locally/) for the full list of builds. Install libtorch before tinytorch so the backend is ready on the first install:

```bash
mkdir -p ~/.local/lib

# CUDA 12.8 (recommended)
curl -sL "https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Bcu128.zip" \
  -o /tmp/libtorch.zip

# CUDA 12.6
curl -sL "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Bcu126.zip" \
  -o /tmp/libtorch.zip

# CPU only
curl -sL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Bcpu.zip" \
  -o /tmp/libtorch.zip

unzip -q /tmp/libtorch.zip -d ~/.local/lib
```

Or from R (requires tinytorch to already be installed in stub mode):

```r
tinytorch::install_libtorch()  # CPU only
# Then reinstall to pick up the backend
install.packages("tinytorch")
```

Both methods install to `~/.local/lib/libtorch`. You can also set `LIBTORCH_HOME` to point at a custom location.

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

### What's missing on Linux

Some libtorch ops only ship with Apple MPS kernels (e.g. `_fused_rms_norm` in 2.8). These have no CPU or CUDA implementation, so tinytorch doesn't bind them. We track this across releases and will pick them up when PyTorch adds CUDA/CPU dispatch.

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
