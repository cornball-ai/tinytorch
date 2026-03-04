if (!Rtorch::is_available()) exit_file("LibTorch not available")

# test_analysis.R - Tests for analysis passes

# ============================================================
# broadcast_shapes
# ============================================================

# Same shapes
expect_equal(broadcast_shapes(c(3L, 4L), c(3L, 4L)), c(3L, 4L))

# One is 1
expect_equal(broadcast_shapes(c(3L, 1L), c(3L, 4L)), c(3L, 4L))
expect_equal(broadcast_shapes(c(1L, 4L), c(3L, 4L)), c(3L, 4L))

# Different ranks - right-aligns
expect_equal(broadcast_shapes(c(4L), c(3L, 4L)), c(3L, 4L))
expect_equal(broadcast_shapes(c(3L, 4L), c(4L)), c(3L, 4L))

# Batch broadcast
expect_equal(broadcast_shapes(c(2L, 3L, 4L), c(3L, 4L)), c(2L, 3L, 4L))
expect_equal(broadcast_shapes(c(1L, 3L, 4L), c(2L, 3L, 4L)), c(2L, 3L, 4L))

# Scalar broadcast
expect_equal(broadcast_shapes(1L, c(3L, 4L)), c(3L, 4L))

# Incompatible
expect_error(broadcast_shapes(c(3L, 4L), c(5L, 4L)), "Incompatible")

# NULL propagation
expect_null(broadcast_shapes(NULL, c(3L, 4L)))
expect_null(broadcast_shapes(c(3L, 4L), NULL))

# ============================================================
# Shape inference: elementwise unary
# ============================================================

ir_unary <- lower_to_ir(list(quote(x$relu()$sigmoid())))
ir_shaped <- infer_shapes(ir_unary, list(x = c(128L, 64L)))

# Input shape propagates through relu and sigmoid
out_node <- ir_shaped$nodes[[as.character(ir_shaped$output_ids[1])]]
expect_equal(out_node$attrs$output_shape, c(128L, 64L))
expect_equal(out_node$attrs$output_dtype, "float32")

# Input node has shape too
input_node <- ir_shaped$nodes[[as.character(ir_shaped$input_ids[1])]]
expect_equal(input_node$attrs$output_shape, c(128L, 64L))

# ============================================================
# Shape inference: binary ops with broadcasting
# ============================================================

ir_add <- lower_to_ir(list(quote(x + y)))
ir_add_shaped <- infer_shapes(ir_add,
  list(x = c(32L, 64L), y = c(1L, 64L)))

out_add <- ir_add_shaped$nodes[[as.character(ir_add_shaped$output_ids[1])]]
expect_equal(out_add$attrs$output_shape, c(32L, 64L))
expect_equal(out_add$attrs$output_dtype, "float32")

# ============================================================
# Shape inference: matmul
# ============================================================

ir_mm <- lower_to_ir(list(quote(x$matmul(y))))
ir_mm_shaped <- infer_shapes(ir_mm,
  list(x = c(128L, 64L), y = c(64L, 32L)))

out_mm <- ir_mm_shaped$nodes[[as.character(ir_mm_shaped$output_ids[1])]]
expect_equal(out_mm$attrs$output_shape, c(128L, 32L))
expect_equal(out_mm$attrs$output_dtype, "float32")

# ============================================================
# Shape inference: batched matmul
# ============================================================

ir_bmm <- lower_to_ir(list(quote(x$bmm(y))))
ir_bmm_shaped <- infer_shapes(ir_bmm,
  list(x = c(4L, 8L, 16L), y = c(4L, 16L, 32L)))

out_bmm <- ir_bmm_shaped$nodes[[as.character(ir_bmm_shaped$output_ids[1])]]
expect_equal(out_bmm$attrs$output_shape, c(4L, 8L, 32L))

# ============================================================
# Shape inference: chain matmul -> relu -> sigmoid
# ============================================================

ir_chain <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu()$sigmoid())
))
ir_chain_shaped <- infer_shapes(ir_chain,
  list(x = c(128L, 64L), y = c(64L, 32L)))

out_chain <- ir_chain_shaped$nodes[[as.character(ir_chain_shaped$output_ids[1])]]
expect_equal(out_chain$attrs$output_shape, c(128L, 32L))
expect_equal(out_chain$attrs$output_dtype, "float32")

# ============================================================
# Shape inference: comparison ops -> bool dtype
# ============================================================

ir_cmp <- lower_to_ir(list(quote(x > y)))
ir_cmp_shaped <- infer_shapes(ir_cmp,
  list(x = c(10L, 20L), y = c(10L, 20L)))

out_cmp <- ir_cmp_shaped$nodes[[as.character(ir_cmp_shaped$output_ids[1])]]
expect_equal(out_cmp$attrs$output_dtype, "bool")
expect_equal(out_cmp$attrs$output_shape, c(10L, 20L))

# ============================================================
# Shape inference: constants get their literal shape
# ============================================================

ir_const <- lower_to_ir(list(quote(x + 1)))
ir_const_shaped <- infer_shapes(ir_const, list(x = c(5L, 5L)))

# Find constant node
const_nodes <- Filter(function(n) n$op == "constant", ir_const_shaped$nodes)
expect_equal(length(const_nodes), 1L)
expect_equal(const_nodes[[1]]$attrs$output_shape, 1L)  # scalar constant
expect_equal(const_nodes[[1]]$attrs$output_dtype, "float32")

# The add result broadcasts [5,5] with [1] -> [5,5]
out_const <- ir_const_shaped$nodes[[as.character(ir_const_shaped$output_ids[1])]]
expect_equal(out_const$attrs$output_shape, c(5L, 5L))

# ============================================================
# Shape inference: dtype propagation
# ============================================================

ir_dtype <- lower_to_ir(list(quote(x$relu())))
ir_dtype_shaped <- infer_shapes(ir_dtype,
  list(x = c(3L, 3L)),
  list(x = "float64"))

out_dtype <- ir_dtype_shaped$nodes[[as.character(ir_dtype_shaped$output_ids[1])]]
expect_equal(out_dtype$attrs$output_dtype, "float64")

# ============================================================
# Shape inference: original graph not mutated
# ============================================================

ir_orig <- lower_to_ir(list(quote(x$relu())))
ir_new <- infer_shapes(ir_orig, list(x = c(3L, 3L)))

# Original should not have shape info
orig_out <- ir_orig$nodes[[as.character(ir_orig$output_ids[1])]]
expect_null(orig_out$attrs$output_shape)

# New should have shape info
new_out <- ir_new$nodes[[as.character(ir_new$output_ids[1])]]
expect_equal(new_out$attrs$output_shape, c(3L, 3L))

# ============================================================
# Shape inference: negation (unary)
# ============================================================

ir_neg <- lower_to_ir(list(quote(-x)))
ir_neg_shaped <- infer_shapes(ir_neg, list(x = c(4L, 8L)))

out_neg <- ir_neg_shaped$nodes[[as.character(ir_neg_shaped$output_ids[1])]]
expect_equal(out_neg$attrs$output_shape, c(4L, 8L))

# ============================================================
# Printer shows shape annotations
# ============================================================

ir_print <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
ir_print_shaped <- infer_shapes(ir_print,
  list(x = c(128L, 64L), y = c(64L, 32L)))

output <- capture.output(print(ir_print_shaped))
expect_true(any(grepl("float32\\[128, 64\\]", output)))
expect_true(any(grepl("float32\\[128, 32\\]", output)))
expect_true(any(grepl("input\\[x\\]", output)))
expect_true(any(grepl("matmul", output)))

# ============================================================
# Liveness: linear chain
# ============================================================

ir_live <- lower_to_ir(list(
  quote(z <- x$relu()),
  quote(z$sigmoid())
))
liveness <- analyze_liveness(ir_live)

# x (input) consumed by relu
# relu consumed by sigmoid
# sigmoid is output -> Inf
out_id <- as.character(ir_live$output_ids[1])
expect_equal(liveness[[out_id]], Inf)

# Input consumed by relu (node id 2)
inp_id <- as.character(ir_live$input_ids[1])
relu_node <- Filter(function(n) n$op == "relu", ir_live$nodes)[[1]]
expect_equal(liveness[[inp_id]], relu_node$id)

# ============================================================
# Liveness: shared input
# ============================================================

ir_shared <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
liveness_shared <- analyze_liveness(ir_shared)

# x is used by both relu and sigmoid - last_use is the later one
inp_x <- as.character(ir_shared$input_ids[1])
relu_n <- Filter(function(n) n$op == "relu", ir_shared$nodes)[[1]]
sig_n <- Filter(function(n) n$op == "sigmoid", ir_shared$nodes)[[1]]
expect_equal(liveness_shared[[inp_x]], max(relu_n$id, sig_n$id))

# ============================================================
# Liveness: dead code
# ============================================================

# Build a graph manually where a node has no consumers
dead_graph <- Rtorch:::ir_graph(
  nodes = list(
    "1" = Rtorch:::ir_node(1L, "input", attrs = list(name = "x")),
    "2" = Rtorch:::ir_node(2L, "relu", inputs = 1L),
    "3" = Rtorch:::ir_node(3L, "sigmoid", inputs = 1L)
  ),
  input_ids = 1L,
  output_ids = 3L  # relu (node 2) is dead
)
liveness_dead <- analyze_liveness(dead_graph)
expect_true(is.na(liveness_dead[["2"]]))   # relu is dead
expect_equal(liveness_dead[["3"]], Inf)     # sigmoid is output

# ============================================================
# Liveness: output nodes -> Inf
# ============================================================

ir_out <- lower_to_ir(list(quote(x$relu())))
liveness_out <- analyze_liveness(ir_out)
out_id_str <- as.character(ir_out$output_ids[1])
expect_equal(liveness_out[[out_id_str]], Inf)

# ============================================================
# Dependency: linear chain
# ============================================================

ir_dep <- lower_to_ir(list(
  quote(z <- x$relu()),
  quote(z$sigmoid())
))
deps <- analyze_deps(ir_dep)

# Input has no deps
inp_str <- as.character(ir_dep$input_ids[1])
expect_equal(length(deps[[inp_str]]), 0L)

# relu depends on input
relu_dep <- Filter(function(n) n$op == "relu", ir_dep$nodes)[[1]]
expect_true(ir_dep$input_ids[1] %in% deps[[as.character(relu_dep$id)]])

# sigmoid depends on input and relu
sig_dep <- Filter(function(n) n$op == "sigmoid", ir_dep$nodes)[[1]]
sig_deps <- deps[[as.character(sig_dep$id)]]
expect_true(ir_dep$input_ids[1] %in% sig_deps)
expect_true(relu_dep$id %in% sig_deps)

# ============================================================
# Dependency: independent branches
# ============================================================

ir_branch <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- y$sigmoid()),
  quote(a + b)
))
deps_branch <- analyze_deps(ir_branch)

# relu depends on x, sigmoid depends on y - no overlap
x_id <- NULL
y_id <- NULL
for (n in ir_branch$nodes) {
  if (n$op == "input" && n$attrs$name == "x") x_id <- n$id
  if (n$op == "input" && n$attrs$name == "y") y_id <- n$id
}
relu_br <- Filter(function(n) n$op == "relu", ir_branch$nodes)[[1]]
sig_br <- Filter(function(n) n$op == "sigmoid", ir_branch$nodes)[[1]]

relu_deps <- deps_branch[[as.character(relu_br$id)]]
sig_deps <- deps_branch[[as.character(sig_br$id)]]
expect_equal(length(intersect(relu_deps, sig_deps)), 0L)

# ============================================================
# Dependency: diamond pattern
# ============================================================

# x -> relu -> add
# x -> sigmoid -> add
# Both branches share x as dependency
ir_diamond <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
deps_diamond <- analyze_deps(ir_diamond)

add_node <- Filter(function(n) n$op == "add", ir_diamond$nodes)[[1]]
add_deps <- deps_diamond[[as.character(add_node$id)]]

# add should depend on x (the shared input)
x_id_d <- ir_diamond$input_ids[1]
expect_true(x_id_d %in% add_deps)

# add should depend on both relu and sigmoid
relu_d <- Filter(function(n) n$op == "relu", ir_diamond$nodes)[[1]]
sig_d <- Filter(function(n) n$op == "sigmoid", ir_diamond$nodes)[[1]]
expect_true(relu_d$id %in% add_deps)
expect_true(sig_d$id %in% add_deps)
