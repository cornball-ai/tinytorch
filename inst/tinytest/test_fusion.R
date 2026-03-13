if (!tinytorch::is_available()) exit_file("LibTorch not available")

# Tests for automatic kernel fusion in torch_eval

# =============================================================================
# Pattern matching
# =============================================================================

# Test relu+sigmoid fusion detection
x <- torch_randn(100)
result <- torch_eval({ x$relu()$sigmoid() }, x = x, .fuse = TRUE)
expected <- x$relu()$sigmoid()
expect_true(torch_allclose(result, expected),
            info = "torch_eval fuses relu+sigmoid")

# Test relu+sigmoid+tanh fusion detection
result <- torch_eval({ x$relu()$sigmoid()$tanh() }, x = x, .fuse = TRUE)
expected <- x$relu()$sigmoid()$tanh()
expect_true(torch_allclose(result, expected),
            info = "torch_eval fuses relu+sigmoid+tanh")

# Test SiLU pattern: x * x$sigmoid()
result <- torch_eval({ x * x$sigmoid() }, x = x, .fuse = TRUE)
expected <- x * torch_sigmoid(x)
expect_true(torch_allclose(result, expected, atol = 1e-5),
            info = "torch_eval fuses SiLU pattern")

# =============================================================================
# Assignment with fusion
# =============================================================================

# Test fusion with assignment
result <- torch_eval({
  y <- x$relu()$sigmoid()
  y
}, x = x, .fuse = TRUE)
expected <- x$relu()$sigmoid()
expect_true(torch_allclose(result, expected),
            info = "fusion works through assignment")

# =============================================================================
# Mixed fusion + non-fuseable
# =============================================================================

# Test that non-fuseable ops fall through
y <- torch_randn(100)
result <- torch_eval({
  z <- x + y
  z$relu()$sigmoid()
}, x = x, y = y, .fuse = TRUE)
expected <- (x + y)$relu()$sigmoid()
expect_true(torch_allclose(result, expected),
            info = "fusion handles mixed fuseable/non-fuseable")

# =============================================================================
# .fuse=FALSE should skip fusion
# =============================================================================

result <- torch_eval({ x$relu()$sigmoid() }, x = x, .fuse = FALSE)
expected <- x$relu()$sigmoid()
expect_true(torch_allclose(result, expected),
            info = ".fuse=FALSE still produces correct result")

# =============================================================================
# Correctness at different tensor sizes
# =============================================================================

for (n in c(10, 1000, 100000)) {
  x <- torch_randn(n)
  result <- torch_eval({ x$relu()$sigmoid() }, x = x, .fuse = TRUE)
  expected <- x$relu()$sigmoid()
  expect_true(torch_allclose(result, expected),
              info = sprintf("fusion correct at size %d", n))
}
