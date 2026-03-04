if (!Rtorch::is_available()) exit_file("LibTorch not available")

# test_compile.R - Tests for compilation layer

# --- Variable analysis ---

vars1 <- analyze_variables(quote(z <- x$matmul(y)))
expect_true("x" %in% vars1$reads)
expect_true("y" %in% vars1$reads)
expect_true("z" %in% vars1$writes)
expect_true("x" %in% vars1$inputs)
expect_true("y" %in% vars1$inputs)

vars2 <- analyze_variables(quote({
  z <- x$matmul(y)
  w <- z$relu()
}))
expect_true("x" %in% vars2$inputs)
expect_true("y" %in% vars2$inputs)
expect_false("z" %in% vars2$inputs)  # z is written before read
expect_true("z" %in% vars2$writes)
expect_true("w" %in% vars2$writes)

# --- Compiled execution produces correct results ---

x <- torch_randn(3, 3)
y <- torch_randn(3, 3)

# Direct execution
result_direct <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x, y = y, .compile = FALSE)

# Compiled execution
result_compiled <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x, y = y, .compile = TRUE)

# Results should match
expect_inherits(result_compiled, "torch_tensor")
diff <- (result_direct - result_compiled)$abs()$max()$item()
expect_true(diff < 1e-6)

# --- More complex expression ---

a <- torch_randn(32, 64)
b <- torch_randn(64, 64)
c <- torch_randn(32, 64)

result_direct2 <- torch_eval({
  t1 <- a$matmul(b)
  t2 <- t1$add(c)
  t3 <- t2$relu()
  t3$softmax(dim = 2L)
}, a = a, b = b, c = c, .compile = FALSE)

result_compiled2 <- torch_eval({
  t1 <- a$matmul(b)
  t2 <- t1$add(c)
  t3 <- t2$relu()
  t3$softmax(dim = 2L)
}, a = a, b = b, c = c, .compile = TRUE)

diff2 <- (result_direct2 - result_compiled2)$abs()$max()$item()
expect_true(diff2 < 1e-5)

# --- With graph breaks - should still work ---

# Expression with a graph break (unknown function)
# The compiled version should handle this gracefully
my_scale <- 2.0  # R scalar, not tensor

result_break <- torch_eval({
  z <- x$matmul(y)
  z$mul(my_scale)
}, x = x, y = y, .compile = TRUE)

expect_inherits(result_break, "torch_tensor")

# --- Single operation ---

x_single <- torch_randn(5, 5)

result_single <- torch_eval({
  x$relu()
}, x = x_single, .compile = TRUE)

expected_single <- x_single$relu()
diff_single <- (result_single - expected_single)$abs()$max()$item()
expect_true(diff_single < 1e-6)

# --- execute_compiled directly ---

env <- new.env()
env$x <- torch_randn(4, 4)
env$y <- torch_randn(4, 4)

expr <- quote({
  z <- x$matmul(y)
  z$relu()
})

result_exec <- execute_compiled(expr, env, compile = TRUE, verbose = FALSE)
expect_inherits(result_exec, "torch_tensor")

# --- Caching tests ---

# Clear cache first
clear_cache()
stats0 <- cache_stats()
expect_equal(stats0$size, 0)

# First call should compile and cache
x_cache <- torch_randn(4, 4)
y_cache <- torch_randn(4, 4)

result1 <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x_cache, y = y_cache, .compile = TRUE)

stats1 <- cache_stats()
expect_equal(stats1$size, 1)

# Second call with same shapes should use cache
result2 <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x_cache, y = y_cache, .compile = TRUE)

stats2 <- cache_stats()
expect_equal(stats2$size, 1)  # Still 1 - cache hit

# Different shapes should create new cache entry
x_big <- torch_randn(8, 8)
y_big <- torch_randn(8, 8)

result3 <- torch_eval({
  z <- x$matmul(y)
  z$relu()
}, x = x_big, y = y_big, .compile = TRUE)

stats3 <- cache_stats()
expect_equal(stats3$size, 2)  # Now 2 entries

# Results should all be correct
expect_inherits(result1, "torch_tensor")
expect_inherits(result2, "torch_tensor")
expect_inherits(result3, "torch_tensor")

# Clear and verify
n_cleared <- clear_cache()
expect_equal(n_cleared, 2)
expect_equal(cache_stats()$size, 0)
