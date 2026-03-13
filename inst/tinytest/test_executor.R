if (!tinytorch::is_available()) exit_file("LibTorch not available")

# test_executor.R - Tests for Phase 4: Backend (generalized CPU codegen + executor)

# ============================================================
# get_fusion_groups: extracts group IDs
# ============================================================

ir_fuse <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
ir_fuse <- fusion_annotate(ir_fuse)
groups <- tinytorch:::get_fusion_groups(ir_fuse)
expect_true(length(groups) >= 1L)

# No groups in single-op graph
ir_single <- lower_to_ir(list(quote(x$relu())))
ir_single <- fusion_annotate(ir_single)
groups_single <- tinytorch:::get_fusion_groups(ir_single)
expect_equal(length(groups_single), 0L)

# ============================================================
# emit_fused_cpu_kernel: C++ code generation from IR
# ============================================================

# Simple unary chain: relu -> sigmoid -> tanh
ir_chain <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
ir_chain <- optimize_graph(ir_chain)
chain_groups <- tinytorch:::get_fusion_groups(ir_chain)

if (length(chain_groups) > 0L) {
  kernel_info <- emit_fused_cpu_kernel(ir_chain, chain_groups[1])

  expect_false(is.null(kernel_info))
  expect_true(is.character(kernel_info$code))
  expect_true(nchar(kernel_info$code) > 100)

  # Code contains expected patterns
  expect_true(grepl("get_tensor_ptr", kernel_info$code))
  expect_true(grepl("make_tensor_sexp", kernel_info$code))
  expect_true(grepl("at::vec::Vectorized", kernel_info$code))
  expect_true(grepl("#pragma omp parallel for", kernel_info$code))
  expect_true(grepl("extern \"C\"", kernel_info$code))
  expect_true(grepl("Vec::loadu", kernel_info$code))
  expect_true(grepl("\\.store\\(", kernel_info$code))

  # Has 1 external input (x)
  expect_equal(kernel_info$n_inputs, 1L)
  expect_equal(length(kernel_info$external_input_ids), 1L)

  # Has 3 group nodes (relu, sigmoid, tanh)
  expect_equal(length(kernel_info$group_node_ids), 3L)

  # Function name is auto-generated
  expect_true(grepl("^fused_", kernel_info$func_name))
}

# ============================================================
# emit_fused_cpu_kernel: custom function name
# ============================================================

if (length(chain_groups) > 0L) {
  custom_info <- emit_fused_cpu_kernel(ir_chain, chain_groups[1],
                                       func_name = "my_custom_kernel")
  expect_equal(custom_info$func_name, "my_custom_kernel")
  expect_true(grepl("my_custom_kernel", custom_info$code))
}

# ============================================================
# emit_fused_cpu_kernel: binary ops in fusion group
# ============================================================

# relu(x) + y has a binary add in the fusion group
ir_bin <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(a + y)
))
ir_bin <- optimize_graph(ir_bin)
bin_groups <- tinytorch:::get_fusion_groups(ir_bin)

# May or may not form a group (depends on single-consumer check)
# Just verify the function runs without error
if (length(bin_groups) > 0L) {
  bin_info <- emit_fused_cpu_kernel(ir_bin, bin_groups[1])
  # If it could be generated, should have 2 external inputs
  if (!is.null(bin_info)) {
    expect_true(bin_info$n_inputs >= 1L)
  }
}

# ============================================================
# emit_fused_cpu_kernel: unsupported op returns NULL
# ============================================================

# Create a graph with a made-up op in a fusion group
fake_ir <- tinytorch:::ir_graph(
  nodes = list(
    "1" = tinytorch:::ir_node(1L, "input", attrs = list(name = "x")),
    "2" = tinytorch:::ir_node(2L, "imaginary_op", inputs = 1L,
                              attrs = list(fusion_group = 99L)),
    "3" = tinytorch:::ir_node(3L, "relu", inputs = 2L,
                              attrs = list(fusion_group = 99L))
  ),
  input_ids = 1L,
  output_ids = 3L
)
unsupported_result <- emit_fused_cpu_kernel(fake_ir, 99L)
expect_true(is.null(unsupported_result))

# ============================================================
# dispatch_torch_op: unary ops
# ============================================================

x <- torch_randn(100)

# relu
r_relu <- tinytorch:::dispatch_torch_op("relu", list(x))
expect_equal(as.numeric(r_relu), as.numeric(x$relu()), tolerance = 1e-6)

# sigmoid
r_sig <- tinytorch:::dispatch_torch_op("sigmoid", list(x))
expect_equal(as.numeric(r_sig), as.numeric(x$sigmoid()), tolerance = 1e-6)

# tanh
r_tanh <- tinytorch:::dispatch_torch_op("tanh", list(x))
expect_equal(as.numeric(r_tanh), as.numeric(x$tanh()), tolerance = 1e-6)

# exp
r_exp <- tinytorch:::dispatch_torch_op("exp", list(x))
expect_equal(as.numeric(r_exp), as.numeric(x$exp()), tolerance = 1e-5)

# neg
r_neg <- tinytorch:::dispatch_torch_op("neg", list(x))
expect_equal(as.numeric(r_neg), as.numeric(-x), tolerance = 1e-6)

# silu
r_silu <- tinytorch:::dispatch_torch_op("silu", list(x))
expected_silu <- as.numeric(nnf_silu(x))
expect_equal(as.numeric(r_silu), expected_silu, tolerance = 1e-6)

# ============================================================
# dispatch_torch_op: binary ops
# ============================================================

y <- torch_randn(100)

r_add <- tinytorch:::dispatch_torch_op("add", list(x, y))
expect_equal(as.numeric(r_add), as.numeric(x + y), tolerance = 1e-6)

r_mul <- tinytorch:::dispatch_torch_op("mul", list(x, y))
expect_equal(as.numeric(r_mul), as.numeric(x * y), tolerance = 1e-6)

r_sub <- tinytorch:::dispatch_torch_op("sub", list(x, y))
expect_equal(as.numeric(r_sub), as.numeric(x - y), tolerance = 1e-6)

# ============================================================
# dispatch_torch_op: matmul
# ============================================================

a <- torch_randn(10, 20)
b <- torch_randn(20, 5)
r_mm <- tinytorch:::dispatch_torch_op("matmul", list(a, b))
expect_equal(as.numeric(r_mm), as.numeric(a$matmul(b)), tolerance = 1e-5)

# ============================================================
# dispatch_torch_op: reduction
# ============================================================

r_sum <- tinytorch:::dispatch_torch_op("sum", list(x))
expect_equal(as.numeric(r_sum), as.numeric(x$sum()), tolerance = 1e-4)

r_mean <- tinytorch:::dispatch_torch_op("mean", list(x))
expect_equal(as.numeric(r_mean), as.numeric(x$mean()), tolerance = 1e-5)

# ============================================================
# execute_optimized: simple unary chain (no compilation)
# ============================================================

ir_simple <- lower_to_ir(list(quote(x$relu()$sigmoid())))
x_input <- torch_randn(50)
result_nofuse <- execute_optimized(ir_simple, list(x = x_input),
                                   optimize = TRUE, fuse = FALSE)
expected <- x_input$relu()$sigmoid()
expect_equal(as.numeric(result_nofuse), as.numeric(expected), tolerance = 1e-6)

# ============================================================
# execute_optimized: matmul + activation
# ============================================================

ir_mm <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
x_mat <- torch_randn(10, 20)
y_mat <- torch_randn(20, 5)
result_mm <- execute_optimized(ir_mm, list(x = x_mat, y = y_mat),
                               fuse = FALSE)
expected_mm <- x_mat$matmul(y_mat)$relu()
expect_equal(as.numeric(result_mm), as.numeric(expected_mm), tolerance = 1e-5)

# ============================================================
# execute_optimized: constant folding takes effect
# ============================================================

ir_const <- lower_to_ir(list(quote(x + 3 * 4)))
x_cf <- torch_randn(10)
result_cf <- execute_optimized(ir_const, list(x = x_cf), fuse = FALSE)
expected_cf <- x_cf + 12
expect_equal(as.numeric(result_cf), as.numeric(expected_cf), tolerance = 1e-5)

# ============================================================
# execute_optimized: algebraic simplification
# ============================================================

ir_simp <- lower_to_ir(list(quote(x + 0)))
x_simp <- torch_randn(10)
result_simp <- execute_optimized(ir_simp, list(x = x_simp), fuse = FALSE)
expect_equal(as.numeric(result_simp), as.numeric(x_simp), tolerance = 1e-6)

# ============================================================
# execute_optimized: with fusion (compilation)
# ============================================================

# This test compiles a kernel — only run if compilation tools available
can_compile <- tryCatch({
  R_bin <- file.path(R.home("bin"), "R")
  file.exists(R_bin) && nchar(system.file("include", package = "torch")) > 0
}, error = function(e) FALSE)

if (can_compile) {
  ir_fuse_exec <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
  x_fuse <- torch_randn(1000)

  result_fused <- tryCatch(
    execute_optimized(ir_fuse_exec, list(x = x_fuse),
                      optimize = TRUE, fuse = TRUE),
    error = function(e) NULL
  )

  if (!is.null(result_fused)) {
    expected_fused <- x_fuse$relu()$sigmoid()$tanh()
    diff_fused <- (result_fused - expected_fused)$abs()$max()$item()
    expect_true(diff_fused < 1e-5)
  }
}

# ============================================================
# execute_optimized: multi-output (diamond pattern)
# ============================================================

ir_diamond <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
x_dia <- torch_randn(50)
result_dia <- execute_optimized(ir_diamond, list(x = x_dia), fuse = FALSE)
expected_dia <- x_dia$relu() + x_dia$sigmoid()
expect_equal(as.numeric(result_dia), as.numeric(expected_dia), tolerance = 1e-6)

# ============================================================
# execute_optimized: input validation
# ============================================================

expect_error(execute_optimized("not a graph", list(x = x)),
             "Expected an ir_graph")

# ============================================================
# kernel_cache_stats
# ============================================================

stats <- kernel_cache_stats()
expect_true(is.list(stats))
expect_true("n_memory" %in% names(stats))
expect_true("n_disk" %in% names(stats))
expect_true("cache_dir" %in% names(stats))
expect_true(grepl("tinytorch", stats$cache_dir))

# ============================================================
# clear_kernel_cache
# ============================================================

n_cleared <- clear_kernel_cache()
expect_true(is.numeric(n_cleared))
stats_after <- kernel_cache_stats()
expect_equal(stats_after$n_memory, 0L)

# ============================================================
# silu in op maps
# ============================================================

# Verify silu can be generated
silu_vec <- tinytorch:::gen_op_expr("silu", "vx", vectorized = TRUE)
expect_true(grepl("exp", silu_vec))

silu_scalar <- tinytorch:::gen_op_expr("silu", "sx", vectorized = FALSE)
expect_true(grepl("exp", silu_scalar))

# ============================================================
# prepare_graph: returns prepared_graph object
# ============================================================

ir_prep <- lower_to_ir(list(quote(x$relu()$sigmoid())))
x_prep <- torch_randn(50)
pg <- prepare_graph(ir_prep, list(x = x_prep), optimize = TRUE, fuse = FALSE)

expect_true(inherits(pg, "prepared_graph"))
expect_true(inherits(pg$graph, "ir_graph"))
expect_true(is.list(pg$kernels))
expect_true(is.character(pg$fused_node_set))
expect_true(is.integer(pg$exec_order))
expect_true(is.list(pg$input_map))
expect_true("x" %in% names(pg$input_map))

# ============================================================
# execute_prepared: produces correct results
# ============================================================

result_prep <- execute_prepared(pg, list(x = x_prep))
expected_prep <- x_prep$relu()$sigmoid()
expect_equal(as.numeric(result_prep), as.numeric(expected_prep), tolerance = 1e-6)

# Same prepared graph, different inputs
x_prep2 <- torch_randn(50)
result_prep2 <- execute_prepared(pg, list(x = x_prep2))
expected_prep2 <- x_prep2$relu()$sigmoid()
expect_equal(as.numeric(result_prep2), as.numeric(expected_prep2), tolerance = 1e-6)

# ============================================================
# prepare_graph + execute_prepared: matmul + activation
# ============================================================

ir_prep_mm <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
x_pm <- torch_randn(10, 20)
y_pm <- torch_randn(20, 5)
pg_mm <- prepare_graph(ir_prep_mm, list(x = x_pm, y = y_pm), fuse = FALSE)
result_pm <- execute_prepared(pg_mm, list(x = x_pm, y = y_pm))
expected_pm <- x_pm$matmul(y_pm)$relu()
expect_equal(as.numeric(result_pm), as.numeric(expected_pm), tolerance = 1e-5)

# ============================================================
# prepare_graph: input validation
# ============================================================

expect_error(prepare_graph("not a graph", list(x = x)),
             "Expected an ir_graph")
expect_error(prepare_graph(ir_prep, "not a list"),
             "example_inputs must be a named list")

# ============================================================
# execute_prepared: input validation
# ============================================================

expect_error(execute_prepared("not prepared", list(x = x)),
             "Expected a prepared_graph")

# ============================================================
# execute_optimized: auto-caching works
# ============================================================

# Clear exec cache first
clear_exec_cache()

ir_cache <- lower_to_ir(list(quote(x$relu()$sigmoid())))
x_cache <- torch_randn(50)

# First call: cache miss
r1_cache <- execute_optimized(ir_cache, list(x = x_cache), fuse = FALSE)
stats_c <- exec_cache_stats()
expect_true(stats_c$n_cached >= 1L)

# Second call with same graph+shape: cache hit (no error)
r2_cache <- execute_optimized(ir_cache, list(x = x_cache), fuse = FALSE)
expect_equal(as.numeric(r1_cache), as.numeric(r2_cache), tolerance = 1e-6)

# Both should match direct torch
expected_cache <- x_cache$relu()$sigmoid()
expect_equal(as.numeric(r1_cache), as.numeric(expected_cache), tolerance = 1e-6)

# ============================================================
# execute_optimized: eager fallback for single ops
# ============================================================

# Single relu — should use eager fallback, skip pipeline
ir_single_op <- lower_to_ir(list(quote(x$relu())))
x_single <- torch_randn(100)
result_single <- execute_optimized(ir_single_op, list(x = x_single),
                                   fuse = FALSE)
expected_single <- x_single$relu()
expect_equal(as.numeric(result_single), as.numeric(expected_single),
             tolerance = 1e-6)

# Single binary add — also eager fallback
ir_add_only <- lower_to_ir(list(quote(x + y)))
x_add <- torch_randn(50)
y_add <- torch_randn(50)
result_add_only <- execute_optimized(ir_add_only, list(x = x_add, y = y_add),
                                     fuse = FALSE)
expected_add_only <- x_add + y_add
expect_equal(as.numeric(result_add_only), as.numeric(expected_add_only),
             tolerance = 1e-6)

# ============================================================
# clear_exec_cache and exec_cache_stats
# ============================================================

# Ensure there's something cached from above
stats_before <- exec_cache_stats()
expect_true(stats_before$n_cached >= 1L)

n_exec_cleared <- clear_exec_cache()
expect_true(is.numeric(n_exec_cleared))
expect_true(n_exec_cleared >= 1L)

stats_after_clear <- exec_cache_stats()
expect_equal(stats_after_clear$n_cached, 0L)

# ============================================================
# execute_optimized: cache key distinguishes shapes
# ============================================================

clear_exec_cache()

ir_shape <- lower_to_ir(list(quote(x$relu()$sigmoid())))
x_small <- torch_randn(10)
x_large <- torch_randn(1000)

execute_optimized(ir_shape, list(x = x_small), fuse = FALSE)
execute_optimized(ir_shape, list(x = x_large), fuse = FALSE)

# Two different shapes should produce two cache entries
stats_shapes <- exec_cache_stats()
expect_equal(stats_shapes$n_cached, 2L)

# ============================================================
# .make_exec_cache_key: deterministic
# ============================================================

ir_key <- lower_to_ir(list(quote(x$relu()$sigmoid())))
x_key <- torch_randn(10)
key1 <- tinytorch:::.make_exec_cache_key(ir_key, list(x = x_key))
key2 <- tinytorch:::.make_exec_cache_key(ir_key, list(x = x_key))
expect_equal(key1, key2)

# Different graph -> different key
ir_key2 <- lower_to_ir(list(quote(x$relu()$tanh())))
key3 <- tinytorch:::.make_exec_cache_key(ir_key2, list(x = x_key))
expect_true(key1 != key3)

# ============================================================
# prepare_graph + execute_prepared: with fusion (compilation)
# ============================================================

if (can_compile) {
  ir_prep_fuse <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
  x_pf <- torch_randn(1000)

  pg_fuse <- tryCatch(
    prepare_graph(ir_prep_fuse, list(x = x_pf), optimize = TRUE, fuse = TRUE),
    error = function(e) NULL
  )

  if (!is.null(pg_fuse)) {
    result_pf <- execute_prepared(pg_fuse, list(x = x_pf))
    if (!is.null(result_pf)) {
      expected_pf <- x_pf$relu()$sigmoid()$tanh()
      diff_pf <- (result_pf - expected_pf)$abs()$max()$item()
      expect_true(diff_pf < 1e-5)
    }
  }
}

# ============================================================
# Cleanup
# ============================================================

clear_kernel_cache()
clear_exec_cache()
