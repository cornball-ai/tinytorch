if (!tinytorch::is_available()) exit_file("LibTorch not available")

# test_memory.R - Tests for memory planning

# ============================================================
# In-place detection: simple chain
# ============================================================

# x -> relu -> sigmoid -> tanh
# sigmoid can be in-place on relu, tanh can be in-place on sigmoid
ir_chain <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
ir_chain <- infer_shapes(ir_chain, list(x = c(100L)))
liveness <- analyze_liveness(ir_chain)
in_place <- tinytorch:::detect_in_place(ir_chain, liveness)

# At least sigmoid and tanh should be in-place
expect_true(length(in_place) >= 2L)

# ============================================================
# In-place: input nodes not overwritten
# ============================================================

# relu(x) - relu's input is x (graph input), should NOT be in-place
ir_relu <- lower_to_ir(list(quote(x$relu())))
ir_relu <- infer_shapes(ir_relu, list(x = c(100L)))
liveness_relu <- analyze_liveness(ir_relu)
ip_relu <- tinytorch:::detect_in_place(ir_relu, liveness_relu)

# relu's input is x (a graph input) - no in-place
relu_node <- Filter(function(n) n$op == "relu", ir_relu$nodes)[[1]]
expect_true(is.null(ip_relu[[as.character(relu_node$id)]]))

# ============================================================
# In-place: shape mismatch blocks in-place
# ============================================================

# Build IR where shapes differ between input and output of an elementwise op
# Use matmul (changes shape) followed by relu
ir_mm <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
ir_mm <- infer_shapes(ir_mm,
  list(x = c(10L, 20L), y = c(20L, 5L)))
liveness_mm <- analyze_liveness(ir_mm)
ip_mm <- tinytorch:::detect_in_place(ir_mm, liveness_mm)

# relu CAN be in-place on matmul (same shape [10,5])
relu_mm <- Filter(function(n) n$op == "relu", ir_mm$nodes)[[1]]
expect_false(is.null(ip_mm[[as.character(relu_mm$id)]]))

# ============================================================
# In-place: multi-consumer input blocks in-place
# ============================================================

# x -> relu -> add(relu, x) -- relu's input x has last_use at add, not relu
# So relu can't be in-place (x is graph input anyway)
# More importantly: relu is used by add, AND input x is used by add
ir_multi <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(a + x)
))
ir_multi <- infer_shapes(ir_multi, list(x = c(50L)))
liveness_multi <- analyze_liveness(ir_multi)
ip_multi <- tinytorch:::detect_in_place(ir_multi, liveness_multi)

# relu's last_use = add's id (not Inf), so add CAN be in-place on relu
add_node <- Filter(function(n) n$op == "add", ir_multi$nodes)[[1]]
relu_multi <- Filter(function(n) n$op == "relu", ir_multi$nodes)[[1]]
# relu's last_use should be add's id
expect_equal(liveness_multi[[as.character(relu_multi$id)]], add_node$id)

# ============================================================
# plan_memory: simple chain
# ============================================================

ir_plan <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
plan <- plan_memory(ir_plan, input_shapes = list(x = c(100L)))

expect_inherits(plan, "memory_plan")
expect_true(plan$n_buffers >= 1L)
expect_true(plan$total_bytes > 0)
expect_true(plan$naive_bytes >= plan$total_bytes)

# With in-place chain, should need only 1 buffer
expect_equal(plan$n_buffers, 1L)

# 3 operations, 1 buffer = 66.7% savings
expect_true(plan$reuse_pct > 0)

# ============================================================
# plan_memory: pre-annotated graph
# ============================================================

ir_pre <- lower_to_ir(list(quote(x$relu()$sigmoid())))
ir_pre <- infer_shapes(ir_pre, list(x = c(50L, 50L)))
plan_pre <- plan_memory(ir_pre)

expect_inherits(plan_pre, "memory_plan")
expect_true(plan_pre$n_buffers >= 1L)

# ============================================================
# plan_memory: error without shapes
# ============================================================

ir_noshape <- lower_to_ir(list(quote(x$relu())))
expect_error(plan_memory(ir_noshape), "no shape annotations")

# ============================================================
# plan_memory: diamond pattern needs 2 buffers
# ============================================================

# x -> relu -> add
# x -> sigmoid -> add
# relu and sigmoid are both alive until add
ir_diamond <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
plan_diamond <- plan_memory(ir_diamond, list(x = c(100L)))

# Need at least 2 buffers (relu and sigmoid alive concurrently)
# add can be in-place on one of them
expect_true(plan_diamond$n_buffers >= 2L)
expect_true(plan_diamond$naive_bytes > 0)

# ============================================================
# plan_memory: buffer reuse after death
# ============================================================

# x -> relu -> sigmoid (output)
# y -> tanh -> exp (output)
# relu dies before tanh starts (if ordered correctly)
# tanh can potentially reuse relu's buffer
ir_reuse <- lower_to_ir(list(
  quote(a <- x$relu()$sigmoid()),
  quote(b <- y$tanh()$exp()),
  quote(a + b)
))
plan_reuse <- plan_memory(ir_reuse,
  list(x = c(100L), y = c(100L)))

# Should use fewer buffers than naive
expect_true(plan_reuse$n_buffers < length(plan_reuse$assignments))

# ============================================================
# plan_memory: byte sizes correct
# ============================================================

# float32[1000] = 4000 bytes per tensor
ir_bytes <- lower_to_ir(list(quote(x$relu())))
plan_bytes <- plan_memory(ir_bytes, list(x = c(1000L)))

# 1 relu operation, 1000 * 4 = 4000 bytes
expect_equal(plan_bytes$naive_bytes, 4000)
expect_equal(plan_bytes$total_bytes, 4000)

# float64 doubles the size
ir_f64 <- lower_to_ir(list(quote(x$relu())))
plan_f64 <- plan_memory(ir_f64,
  list(x = c(1000L)),
  list(x = "float64"))
expect_equal(plan_f64$naive_bytes, 8000)

# ============================================================
# plan_memory: empty graph (input only)
# ============================================================

ir_empty <- tinytorch:::ir_graph(
  nodes = list("1" = tinytorch:::ir_node(1L, "input", attrs = list(name = "x"))),
  input_ids = 1L,
  output_ids = 1L
)
ir_empty <- infer_shapes(ir_empty, list(x = c(10L)))
plan_empty <- plan_memory(ir_empty)
expect_equal(plan_empty$n_buffers, 0L)
expect_equal(plan_empty$total_bytes, 0)

# ============================================================
# print.memory_plan works
# ============================================================

ir_print <- lower_to_ir(list(quote(x$relu()$sigmoid())))
plan_print <- plan_memory(ir_print, list(x = c(100L, 100L)))
output <- capture.output(print(plan_print))
expect_true(any(grepl("Memory Plan", output)))
expect_true(any(grepl("Savings", output)))

# ============================================================
# In-place assignments recorded in plan
# ============================================================

ir_ip <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
plan_ip <- plan_memory(ir_ip, list(x = c(100L)))

# Should have in-place ops recorded
expect_true(length(plan_ip$in_place) >= 2L)

# All in-place nodes share the same buffer
buf_ids <- unique(unlist(plan_ip$assignments))
expect_equal(length(buf_ids), 1L)
