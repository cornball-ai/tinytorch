# Tests for pre-compiled fused SIMD kernels

# =============================================================================
# fused_relu
# =============================================================================

# Test fused_relu produces correct output
x <- torch_randn(100)
y <- fused_relu(x)
expected <- x$relu()
expect_true(torch_allclose(y, expected),
            info = "fused_relu matches torch relu")

# Test with negative values
x <- torch_tensor(c(-1, -2, 0, 1, 2))
y <- fused_relu(x)
expected <- torch_tensor(c(0, 0, 0, 1, 2))
expect_true(torch_allclose(y, expected),
            info = "fused_relu zeros out negative values")

# Test output class
expect_true(inherits(y, "torch_tensor"),
            info = "fused_relu returns torch_tensor")

# Test error on non-tensor input
expect_error(fused_relu(c(1, 2, 3)),
             info = "fused_relu errors on non-tensor")

# =============================================================================
# fused_relu_sigmoid
# =============================================================================

# Test fused_relu_sigmoid produces correct output
x <- torch_randn(100)
y <- fused_relu_sigmoid(x)
expected <- x$relu()$sigmoid()
expect_true(torch_allclose(y, expected),
            info = "fused_relu_sigmoid matches torch relu+sigmoid")

# Test output range is [0.5, 1] (sigmoid of non-negative values)
expect_true(all(as.numeric(y) >= 0.5),
            info = "fused_relu_sigmoid output >= 0.5")

# =============================================================================
# fused_relu_sigmoid_tanh
# =============================================================================

# Test fused_relu_sigmoid_tanh produces correct output
x <- torch_randn(100)
y <- fused_relu_sigmoid_tanh(x)
expected <- x$relu()$sigmoid()$tanh()
expect_true(torch_allclose(y, expected),
            info = "fused_relu_sigmoid_tanh matches torch relu+sigmoid+tanh")

# Test with larger tensor
x <- torch_randn(10000)
y <- fused_relu_sigmoid_tanh(x)
expected <- x$relu()$sigmoid()$tanh()
expect_true(torch_allclose(y, expected),
            info = "fused_relu_sigmoid_tanh works on larger tensors")

# =============================================================================
# Shape preservation
# =============================================================================

# Test 2D tensor shape preservation
x <- torch_randn(c(10, 20))
y <- fused_relu(x)
expect_equal(dim(y), dim(x),
             info = "fused_relu preserves 2D shape")

# Test 3D tensor shape preservation
x <- torch_randn(c(5, 10, 20))
y <- fused_relu_sigmoid(x)
expect_equal(dim(y), dim(x),
             info = "fused_relu_sigmoid preserves 3D shape")

# =============================================================================
# Tier 1 kernels
# =============================================================================

# Test fused_silu (x * sigmoid(x))
x <- torch_randn(100)
y <- fused_silu(x)
expected <- x * torch_sigmoid(x)
expect_true(torch_allclose(y, expected, atol = 1e-5),
            info = "fused_silu matches x * sigmoid(x)")

# Test fused_gelu (tanh approximation)
x <- torch_randn(100)
y <- fused_gelu(x)
# fused_gelu uses tanh approximation, compare against exact GELU with wider tolerance
expect_true(torch_allclose(y, nnf_gelu(x), atol = 1e-2),
            info = "fused_gelu approximately matches nnf_gelu")

# Test fused_sincos
x <- torch_randn(100)
result <- fused_sincos(x)
expect_true(torch_allclose(result$sin, torch_sin(x), atol = 1e-5),
            info = "fused_sincos sin matches torch_sin")
expect_true(torch_allclose(result$cos, torch_cos(x), atol = 1e-5),
            info = "fused_sincos cos matches torch_cos")

# Test fused_softcap
x <- torch_randn(100)
cap <- 50.0
y <- fused_softcap(x, cap)
expected <- torch_tanh(x / cap) * cap
expect_true(torch_allclose(y, expected, atol = 1e-5),
            info = "fused_softcap matches tanh(x/cap)*cap")

# Test fused_rmsnorm
x <- torch_randn(c(4, 64))
weight <- torch_ones(64)
eps <- 1e-6
y <- fused_rmsnorm(x, weight, eps)
variance <- x$pow(2)$mean(dim = -1L, keepdim = TRUE)
expected <- x * torch_rsqrt(variance + eps) * weight
expect_true(torch_allclose(y, expected, atol = 1e-4),
            info = "fused_rmsnorm matches manual implementation")
