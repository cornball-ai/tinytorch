# Tests for GPU Triton kernel fusion via compile()
# Requires CUDA GPU and ariel package with Triton support.

if (!Rtorch::cuda_is_available() ||
    !requireNamespace("ariel", quietly = TRUE)) exit_file("No CUDA or ariel")

# =============================================================================
# Elementwise fusion on GPU
# =============================================================================

x <- torch_randn(c(100, 200), device = "cuda")

# relu + sigmoid chain (2-op)
compiled <- compile(function(x) x$relu()$sigmoid(), x = x)
expect_true(!is.null(compiled$prepared),
            info = "compile() produces prepared graph")
expect_equal(compiled$prepared$backend, "gpu",
             info = "GPU backend detected")
expect_true(length(compiled$prepared$kernels) > 0,
            info = "At least one fusion kernel compiled")

result <- compiled$fn(x = x)
expected <- x$relu()$sigmoid()
expect_true(torch_allclose(result, expected, atol = 1e-6),
            info = "GPU fused relu+sigmoid matches unfused")

# Verify it's a Triton GPU kernel
k <- compiled$prepared$kernels[[1]]
expect_true(isTRUE(k$gpu), info = "Kernel is GPU (Triton)")

# exp + neg chain
x2 <- torch_randn(c(50, 30), device = "cuda")
compiled2 <- compile(function(x) (-x)$exp(), x = x2)
result2 <- compiled2$fn(x = x2)
expected2 <- (-x2)$exp()
expect_true(torch_allclose(result2, expected2, atol = 1e-6),
            info = "GPU fused neg+exp matches unfused")

# 3-op chain with tanh (decomposed compound op)
compiled3 <- compile(function(x) x$relu()$sigmoid()$tanh(), x = x)
k3 <- compiled3$prepared$kernels[[1]]
expect_true(isTRUE(k3$gpu), info = "3-op chain compiles to GPU kernel")
result3 <- k3$call_fn(x)
expected3 <- x$relu()$sigmoid()$tanh()
expect_true(torch_allclose(result3, expected3, atol = 1e-6),
            info = "GPU fused relu+sigmoid+tanh matches unfused")

# 5-op chain
compiled5 <- compile(function(x) x$relu()$sigmoid()$tanh()$sigmoid()$relu(), x = x)
k5 <- compiled5$prepared$kernels[[1]]
expect_true(isTRUE(k5$gpu), info = "5-op chain compiles to GPU kernel")
result5 <- k5$call_fn(x)
expected5 <- x$relu()$sigmoid()$tanh()$sigmoid()$relu()
expect_true(torch_allclose(result5, expected5, atol = 1e-6),
            info = "GPU fused 5-op chain matches unfused")

# Large tensor
x_large <- torch_randn(c(1024, 1024), device = "cuda")
compiled_l <- compile(function(x) x$relu()$sigmoid(), x = x_large)
result_l <- compiled_l$fn(x = x_large)
expected_l <- x_large$relu()$sigmoid()
expect_true(torch_allclose(result_l, expected_l, atol = 1e-6),
            info = "GPU fusion works on large tensors")
expect_equal(as.integer(result_l$shape), c(1024L, 1024L),
             info = "Output shape preserved for large tensor")

# =============================================================================
# GPU kernel cache (CUDA driver API side)
# =============================================================================

stats <- gpu_kernel_cache_stats()
expect_true(stats$n_cached > 0,
            info = "GPU kernel cache has entries after launch")

# =============================================================================
# PTX disk cache (ariel side)
# =============================================================================

ptx_stats <- ariel::ptx_cache_stats()
expect_true(ptx_stats$n_cached > 0,
            info = "PTX disk cache has entries after compilation")

# =============================================================================
# numel method
# =============================================================================

t <- torch_randn(c(3, 4, 5), device = "cuda")
expect_equal(t$numel(), 60L, info = "numel returns product of shape")
