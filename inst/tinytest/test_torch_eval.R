if (!tinytorch::is_available()) exit_file("LibTorch not available")

# test_torch_eval.R - Tests for torch_eval()

# --- Basic expression capture and evaluation ---

# Simple matmul
x <- torch_randn(3, 3)
y <- torch_randn(3, 3)

result <- torch_eval({
  x$matmul(y)
}, x = x, y = y)

expect_inherits(result, "torch_tensor")
expect_equal(as.numeric(result$shape), c(3, 3))

# Compare to direct evaluation
direct <- x$matmul(y)
expect_equal(as.array(result), as.array(direct))

# --- Multi-statement expressions ---

result2 <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x, y = y)

expect_inherits(result2, "torch_tensor")

# --- Arithmetic operators on tensors ---

a <- torch_tensor(c(1, 2, 3))
b <- torch_tensor(c(4, 5, 6))

result3 <- torch_eval({
  a + b
}, a = a, b = b)

expect_equal(as.array(result3), c(5, 7, 9))

result4 <- torch_eval({
  a * b
}, a = a, b = b)

expect_equal(as.array(result4), c(4, 10, 18))

# --- Using torch_* functions inside expression ---

result5 <- torch_eval({
  torch_sum(x)
}, x = torch_tensor(c(1, 2, 3, 4)))

expect_equal(as.numeric(result5), 10)

# --- Error handling ---

# Unnamed tensor arguments should error
expect_error(
torch_eval({ x + y }, torch_randn(2), torch_randn(2)),
  pattern = "must be named"
)

# Non-tensor arguments should error
expect_error(
  torch_eval({ x + 1 }, x = c(1, 2, 3)),
  pattern = "must be a torch_tensor"
)

# --- Environment resolution ---

# Should resolve non-tensor symbols from parent environment
multiplier <- 2
t <- torch_tensor(c(1, 2, 3))

result6 <- torch_eval({
  t * multiplier
}, t = t)

expect_equal(as.array(result6), c(2, 4, 6))
