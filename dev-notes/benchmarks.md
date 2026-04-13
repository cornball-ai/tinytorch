---
title: "Benchmarks: tinytorch vs torch vs Python"
---

Benchmarks
----------

All benchmarks use libtorch 2.11.0 on CPU. Lower is better.

Three implementations compared:

- **Python** -- `torch 2.11.0+cpu` via pip (CPython 3.10)
- **R torch** -- `torch 0.16.3.9000` (Rcpp + lantern + R7 dispatch)
- **tinytorch** -- Rcpp + S3, direct to libtorch (no lantern, no R7)


### Micro-ops (microseconds per call, 10k iterations)

| Test                         | Python | R torch | tinytorch | speedup vs torch |
|------------------------------|--------|---------|-----------|------------------|
| Function add (10x10)         | 0.5    | 10.1    | 1.3       | 7.8x             |
| Method $add (10x10)          | 0.5    | 13.3    | 3.2       | 4.2x             |
| Chained matmul (10x10)       | 1.3    | 19.1    | 3.4       | 5.6x             |
| Method chain $add()$mul()    | 0.9    | 24.9    | 5.3       | 4.7x             |
| Creation randn(10,10)        | 0.8    | 16.9    | 1.7       | 9.9x             |
| Large matmul (1000x1000)     | 1,371  | 175,545 | 176,005   | 1.0x             |

All times in microseconds.


### Training loop (microseconds per step, 1k iterations)

Reproduces the scenario from [mlverse/torch#268](https://github.com/mlverse/torch/issues/268)
and [#970](https://github.com/mlverse/torch/issues/970): small MLP
(3 layers, batch 32, input dim 20), single forward + backward + Adam step.

| Hidden size | tinytorch | R torch | speedup |
|-------------|-----------|---------|---------|
| 20          | 130       | 1,822   | 14.0x   |
| 40          | 134       | 1,842   | 13.7x   |
| 60          | 306       | 7,230   | 23.6x   |
| 80          | 402       | 2,830   | 7.0x    |
| 100         | 272       | 2,972   | 10.9x   |
| 120         | 498       | 3,026   | 6.1x    |
| 140         | 392       | 3,166   | 8.1x    |
| 160         | 470       | 3,402   | 7.2x    |
| 180         | 428       | 2,972   | 6.9x    |
| 200         | 1,074     | 2,234   | 2.1x    |

**R torch at h=60 shows the GC spike from #970** (7,230 us, 2.5x its neighbors).
tinytorch doesn't exhibit this. The spike in torch comes from R's garbage collector
firing during backward() when the number of intermediate tensors crosses the GC
threshold. tinytorch creates fewer R-side wrapper objects per op, so GC pressure
is lower.

At the fixed benchmark point (h=64, batch=32): **tinytorch is 11x faster** per
training step (250 us vs 2,749 us).


### Dataloader iteration (microseconds per epoch)

1,000 samples, batch size 64, full epoch iteration. Measures the R-side
overhead of getting batches, not the tensor ops inside each batch.

| Implementation | us/epoch | speedup |
|----------------|----------|---------|
| tinytorch      | 390      | --      |
| R torch (coro) | 2,190    | 5.6x   |

torch uses `coro::loop()` with a generator-based dataloader.
tinytorch uses a plain environment with a `.next()` closure.


### Where the overhead lives

```
Python:    function call --> pybind11 --> libtorch                                         ~0.5 us
tinytorch: function call --> .Call()  --> libtorch                                         ~1.4 us
R torch:   function call --> R7 $    --> call_c_function --> Rcpp --> lantern --> libtorch  ~10 us
```

For a single op, the dominant cost is crossing R's .Call() boundary (~1 us).
tinytorch pays this once per op. R torch pays it once per op PLUS R7 class
dispatch, dynamic function resolution, and the lantern dlopen shim.

For a training step (which chains ~15-20 ops), those per-op costs compound.
A 10 us overhead per op * 20 ops = 200 us of pure overhead per step in R torch.
tinytorch's 1.4 us per op * 20 ops = 28 us. The difference is the 11x gap.


### Reproducing

```r
# Micro-ops (requires torch, tinytorch, jsonlite, Python torch venv)
source("bench/benchmark.R")

# Training loop (requires torch and tinytorch)
source("bench/benchmark_trainloop.R")
```
