if (!tinytorch::is_available()) exit_file("LibTorch not available")

# test_codegen.R - Tests for fused kernel code generation

# Skip if Rcpp not available
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  exit_file("Rcpp not available")
}

# --- Test op expression generation ---

# Test that gen_op_expr works (internal function)
relu_vec <- tinytorch:::gen_op_expr("relu", "vx", vectorized = TRUE)
expect_true(grepl("clamp_min", relu_vec))

sigmoid_vec <- tinytorch:::gen_op_expr("sigmoid", "vx", vectorized = TRUE)
expect_true(grepl("exp", sigmoid_vec))

add_vec <- tinytorch:::gen_op_expr("add", c("va", "vb"), vectorized = TRUE)
expect_true(grepl("\\+", add_vec))

# --- Test code generation ---

code <- tinytorch:::gen_fused_kernel_code(
  ops = c("relu", "sigmoid"),
  dtype = "float",
  func_name = "test_kernel"
)
expect_true(grepl("at::vec::Vectorized", code))
expect_true(grepl("test_kernel", code))
expect_true(grepl("#pragma omp parallel", code))

# --- Test fused_ops execution ---

# Simple relu
x <- torch_randn(1000)
result_fused <- fused_ops(x, c("relu"))
result_torch <- x$relu()
diff1 <- (result_fused - result_torch)$abs()$max()$item()
expect_true(diff1 < 1e-6)

# relu + sigmoid chain
result_fused2 <- fused_ops(x, c("relu", "sigmoid"))
result_torch2 <- x$relu()$sigmoid()
diff2 <- (result_fused2 - result_torch2)$abs()$max()$item()
expect_true(diff2 < 1e-6)

# Longer chain
result_fused3 <- fused_ops(x, c("relu", "sigmoid", "tanh"))
result_torch3 <- x$relu()$sigmoid()$tanh()
diff3 <- (result_fused3 - result_torch3)$abs()$max()$item()
expect_true(diff3 < 1e-6)

# --- Test caching ---
# These tests require runtime C++ compilation with ATen headers.
# Skip if compilation is not available (e.g. missing ATen/cpu/vec/vec.h).

clear_codegen_cache()
stats0 <- codegen_cache_stats()
expect_equal(stats0$size, 0)

# Probe: compile one kernel and check if it cached
fused_ops(x, c("exp"))
can_compile <- codegen_cache_stats()$size > 0

if (can_compile) {
  stats1 <- codegen_cache_stats()
  expect_equal(stats1$size, 1)

  # Second call uses cache (should still be 1)
  fused_ops(x, c("exp"))
  stats2 <- codegen_cache_stats()
  expect_equal(stats2$size, 1)

  # Different ops create new entry
  fused_ops(x, c("tanh"))
  stats3 <- codegen_cache_stats()
  expect_equal(stats3$size, 2)

  # Clear
  n_cleared <- clear_codegen_cache()
  expect_equal(n_cleared, 2)

  # --- Test double precision ---

  x_double <- torch_randn(100, dtype = torch_double)
  result_double <- fused_ops(x_double, c("relu", "sigmoid"))
  expected_double <- x_double$relu()$sigmoid()
  diff_double <- (result_double - expected_double)$abs()$max()$item()
  expect_true(diff_double < 1e-10)
}

# --- Cleanup ---
# Rcpp::sourceCpp puts compiled functions in globalenv(), which shadows
# package functions of the same name (fused_relu, fused_relu_sigmoid, etc.)
# Remove them to avoid breaking subsequent tests.
codegen_leftovers <- grep("^fused_", ls(globalenv()), value = TRUE)
if (length(codegen_leftovers) > 0) {
  rm(list = codegen_leftovers, envir = globalenv())
}
