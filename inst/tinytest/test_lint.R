if (!tinytorch::is_available()) exit_file("LibTorch not available")

# test_lint.R - Tests for torch_lint()

# --- Clean expressions ---

result1 <- torch_lint(quote({
  z <- x$matmul(y)
  z$relu()
}))
expect_true(result1$is_clean)
expect_equal(result1$n_breaks, 0L)
expect_equal(result1$n_segments, 1L)

# --- Expressions with breaks ---

result2 <- torch_lint(quote({
  z <- x$matmul(y)
  print(z)
  z$relu()
}))
expect_false(result2$is_clean)
expect_equal(result2$n_breaks, 1L)
expect_equal(result2$n_segments, 3L)
expect_equal(result2$n_graph_segments, 2L)

# Multiple breaks
result3 <- torch_lint(quote({
  z <- x$matmul(y)
  print(z)
  z <- z$relu()
  my_func(z)
  z$softmax(dim = 2L)
}))
expect_false(result3$is_clean)
expect_equal(result3$n_breaks, 2L)

# --- Break types detected ---

# Side effect
result4 <- torch_lint(quote(print(x)))
expect_equal(result4$breaks[[1]]$type, "side_effect")

# Control flow
result5 <- torch_lint(quote(if (cond) x$relu()))
expect_equal(result5$breaks[[1]]$type, "control_flow")

# Unknown function
result6 <- torch_lint(quote(unknown_func(x)))
expect_equal(result6$breaks[[1]]$type, "function_call")

# --- Segment counts ---

# All graph
result7 <- torch_lint(quote({
  a <- x$matmul(y)
  b <- a$relu()
  c <- b$softmax(dim = 2L)
}))
expect_equal(result7$n_segments, 1L)
expect_equal(result7$n_graph_segments, 1L)

# All R code
result8 <- torch_lint(quote({
  print(x)
  message("hello")
}))
expect_equal(result8$n_segments, 1L)
expect_equal(result8$n_graph_segments, 0L)

# --- Single expressions ---

result9 <- torch_lint(quote(x$matmul(y)))
expect_true(result9$is_clean)

result10 <- torch_lint(quote(print(x)))
expect_false(result10$is_clean)
