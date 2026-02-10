# test_optimize.R - Tests for optimization passes

# ============================================================
# Constant Folding
# ============================================================

# Fold 3 * 4 -> 12
ir_cf <- lower_to_ir(list(quote(x + 3 * 4)))
result_cf <- constant_fold(ir_cf)
expect_true(validate_ir(result_cf))

# The mul(3,4) should now be constant(12)
const_nodes <- Filter(function(n) n$op == "constant", result_cf$nodes)
const_vals <- vapply(const_nodes, function(n) {
  v <- n$attrs$value
  if (is.numeric(v) && length(v) == 1) v else NA_real_
}, numeric(1))
expect_true(12 %in% const_vals)

# Non-constant inputs are preserved
add_nodes <- Filter(function(n) n$op == "add", result_cf$nodes)
expect_equal(length(add_nodes), 1L)

# Fold nested: 2 + 3 -> 5
ir_cf2 <- lower_to_ir(list(quote(x + (2 + 3))))
result_cf2 <- constant_fold(ir_cf2)
const_vals2 <- vapply(
  Filter(function(n) n$op == "constant", result_cf2$nodes),
  function(n) {
    v <- n$attrs$value
    if (is.numeric(v) && length(v) == 1) v else NA_real_
  }, numeric(1))
expect_true(5 %in% const_vals2)

# Don't fold when not all inputs are constant
ir_nofold <- lower_to_ir(list(quote(x + y)))
result_nofold <- constant_fold(ir_nofold)
add_nf <- Filter(function(n) n$op == "add", result_nofold$nodes)
expect_equal(length(add_nf), 1L)  # add stays

# ============================================================
# Dead Code Elimination
# ============================================================

# Dead node removed
dead_ir <- Rtorch:::ir_graph(
  nodes = list(
    "1" = Rtorch:::ir_node(1L, "input", attrs = list(name = "x")),
    "2" = Rtorch:::ir_node(2L, "relu", inputs = 1L),
    "3" = Rtorch:::ir_node(3L, "sigmoid", inputs = 1L)
  ),
  input_ids = 1L,
  output_ids = 2L
)

result_dce <- dead_code_eliminate(dead_ir)
expect_true(validate_ir(result_dce))
expect_equal(length(result_dce$nodes), 2L)
expect_null(result_dce$nodes[["3"]])

# Dead input node removed
dead_input <- Rtorch:::ir_graph(
  nodes = list(
    "1" = Rtorch:::ir_node(1L, "input", attrs = list(name = "x")),
    "2" = Rtorch:::ir_node(2L, "input", attrs = list(name = "y")),
    "3" = Rtorch:::ir_node(3L, "relu", inputs = 1L)
  ),
  input_ids = c(1L, 2L),
  output_ids = 3L
)

result_di <- dead_code_eliminate(dead_input)
expect_true(validate_ir(result_di))
expect_equal(length(result_di$input_ids), 1L)
expect_equal(result_di$input_ids, 1L)

# All outputs preserved
ir_live <- lower_to_ir(list(quote(x$relu())))
result_live <- dead_code_eliminate(ir_live)
expect_true(validate_ir(result_live))
expect_equal(length(result_live$nodes), length(ir_live$nodes))

# ============================================================
# Common Subexpression Elimination
# ============================================================

# Two identical matmuls -> one
ir_cse <- lower_to_ir(list(
  quote(a <- x$matmul(y)),
  quote(b <- x$matmul(y)),
  quote(a + b)
))
result_cse <- common_subexpr_eliminate(ir_cse)
expect_true(validate_ir(result_cse))

matmuls <- Filter(function(n) n$op == "matmul", result_cse$nodes)
expect_equal(length(matmuls), 1L)

# The add references the same matmul for both inputs
add_cse <- Filter(function(n) n$op == "add", result_cse$nodes)[[1]]
expect_equal(add_cse$inputs[1], add_cse$inputs[2])

# Different ops not eliminated
ir_diff <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
result_diff <- common_subexpr_eliminate(ir_diff)
relu_n <- Filter(function(n) n$op == "relu", result_diff$nodes)
sig_n <- Filter(function(n) n$op == "sigmoid", result_diff$nodes)
expect_equal(length(relu_n), 1L)
expect_equal(length(sig_n), 1L)

# ============================================================
# Algebraic Simplification: x + 0 -> x
# ============================================================

ir_add0 <- lower_to_ir(list(quote(x + 0)))
result_add0 <- algebraic_simplify(ir_add0)
expect_equal(result_add0$output_ids[1], result_add0$input_ids[1])

# 0 + x -> x (commutative)
ir_0add <- lower_to_ir(list(quote(0 + x)))
result_0add <- algebraic_simplify(ir_0add)
expect_equal(result_0add$output_ids[1], result_0add$input_ids[1])

# ============================================================
# Algebraic Simplification: x * 1 -> x
# ============================================================

ir_mul1 <- lower_to_ir(list(quote(x * 1)))
result_mul1 <- algebraic_simplify(ir_mul1)
expect_equal(result_mul1$output_ids[1], result_mul1$input_ids[1])

# ============================================================
# Algebraic Simplification: x * 0 -> constant(0)
# ============================================================

ir_mul0 <- lower_to_ir(list(quote(x * 0)))
result_mul0 <- algebraic_simplify(ir_mul0)
out_mul0 <- result_mul0$nodes[[as.character(result_mul0$output_ids[1])]]
expect_equal(out_mul0$op, "constant")
expect_equal(out_mul0$attrs$value, 0)

# ============================================================
# Algebraic Simplification: x - x -> constant(0)
# ============================================================

ir_subx <- lower_to_ir(list(quote(x - x)))
result_subx <- algebraic_simplify(ir_subx)
out_subx <- result_subx$nodes[[as.character(result_subx$output_ids[1])]]
expect_equal(out_subx$op, "constant")
expect_equal(out_subx$attrs$value, 0)

# ============================================================
# Algebraic Simplification: x / x -> constant(1)
# ============================================================

ir_divx <- lower_to_ir(list(quote(x / x)))
result_divx <- algebraic_simplify(ir_divx)
out_divx <- result_divx$nodes[[as.character(result_divx$output_ids[1])]]
expect_equal(out_divx$op, "constant")
expect_equal(out_divx$attrs$value, 1)

# ============================================================
# Algebraic Simplification: neg(neg(x)) -> x
# ============================================================

ir_negn <- lower_to_ir(list(quote(-(-x))))
result_negn <- algebraic_simplify(ir_negn)
expect_equal(result_negn$output_ids[1], result_negn$input_ids[1])

# ============================================================
# Algebraic Simplification: relu(relu(x)) -> relu(x)
# ============================================================

ir_relr <- lower_to_ir(list(quote(x$relu()$relu())))
result_relr <- algebraic_simplify(ir_relr)

# Output should point to the first relu, not the second
relu_nodes <- Filter(function(n) n$op == "relu", result_relr$nodes)
first_relu_id <- min(vapply(relu_nodes, function(n) n$id, integer(1)))
expect_equal(result_relr$output_ids[1], first_relu_id)

# ============================================================
# Algebraic Simplification: pow(x, 2) -> mul(x, x)
# ============================================================

ir_pow2 <- lower_to_ir(list(quote(x^2)))
result_pow2 <- algebraic_simplify(ir_pow2)
out_pow2 <- result_pow2$nodes[[as.character(result_pow2$output_ids[1])]]
expect_equal(out_pow2$op, "mul")
expect_equal(out_pow2$inputs[1], out_pow2$inputs[2])  # both are x

# ============================================================
# Algebraic Simplification: pow(x, 0.5) -> sqrt(x)
# ============================================================

ir_sqrt <- lower_to_ir(list(quote(x^0.5)))
result_sqrt <- algebraic_simplify(ir_sqrt)
out_sqrt <- result_sqrt$nodes[[as.character(result_sqrt$output_ids[1])]]
expect_equal(out_sqrt$op, "sqrt")

# ============================================================
# Algebraic Simplification: sigmoid(x) * x -> silu(x)
# ============================================================

ir_silu <- lower_to_ir(list(quote(x$sigmoid() * x)))
result_silu <- algebraic_simplify(ir_silu)
out_silu <- result_silu$nodes[[as.character(result_silu$output_ids[1])]]
expect_equal(out_silu$op, "silu")

# Also x * sigmoid(x)
ir_silu2 <- lower_to_ir(list(quote(x * x$sigmoid())))
result_silu2 <- algebraic_simplify(ir_silu2)
out_silu2 <- result_silu2$nodes[[as.character(result_silu2$output_ids[1])]]
expect_equal(out_silu2$op, "silu")

# ============================================================
# Fusion Annotation: chain gets group
# ============================================================

ir_fuse <- lower_to_ir(list(quote(x$relu()$sigmoid()$tanh())))
result_fuse <- fusion_annotate(ir_fuse)
expect_true(validate_ir(result_fuse))

# All three elementwise ops should have the same fusion_group
ew_nodes <- Filter(function(n) n$op %in% c("relu", "sigmoid", "tanh"),
                   result_fuse$nodes)
groups <- vapply(ew_nodes, function(n) {
  n$attrs$fusion_group %||% NA_integer_
}, integer(1))
expect_true(all(!is.na(groups)))
expect_equal(length(unique(groups)), 1L)  # all same group

# ============================================================
# Fusion Annotation: single op -> no group
# ============================================================

ir_single <- lower_to_ir(list(quote(x$relu())))
result_single <- fusion_annotate(ir_single)
relu_s <- Filter(function(n) n$op == "relu", result_single$nodes)[[1]]
expect_true(is.null(relu_s$attrs$fusion_group))

# ============================================================
# Fusion Annotation: chain broken by non-elementwise
# ============================================================

ir_break <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu()$sigmoid())
))
result_break <- fusion_annotate(ir_break)

# matmul should not be in a group
mm <- Filter(function(n) n$op == "matmul", result_break$nodes)[[1]]
expect_true(is.null(mm$attrs$fusion_group))

# relu + sigmoid should share a group
relu_b <- Filter(function(n) n$op == "relu", result_break$nodes)[[1]]
sig_b <- Filter(function(n) n$op == "sigmoid", result_break$nodes)[[1]]
expect_false(is.null(relu_b$attrs$fusion_group))
expect_equal(relu_b$attrs$fusion_group, sig_b$attrs$fusion_group)

# ============================================================
# optimize_graph: full pipeline
# ============================================================

# Build a graph with multiple optimization opportunities:
# x + (3 * 4) + 0  ->  constant fold 3*4=12, then simplify +0
ir_full <- lower_to_ir(list(quote((x + 3 * 4) + 0)))
result_full <- optimize_graph(ir_full)
expect_true(validate_ir(result_full))

# After optimization: should be add(x, 12) -- the +0 is eliminated
# and 3*4 is folded to 12
add_full <- Filter(function(n) n$op == "add", result_full$nodes)
expect_equal(length(add_full), 1L)

# ============================================================
# optimize_graph: custom passes
# ============================================================

ir_custom <- lower_to_ir(list(quote(x + 0)))
result_custom <- optimize_graph(ir_custom,
  passes = list(algebraic_simplify))
# x + 0 simplified, but dead nodes not cleaned (no DCE in pipeline)
expect_equal(result_custom$output_ids[1], result_custom$input_ids[1])

# ============================================================
# Passes don't break valid graphs
# ============================================================

ir_complex <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(w <- z$relu()$sigmoid()),
  quote(w + z)
))
result_complex <- optimize_graph(ir_complex)
expect_true(validate_ir(result_complex))
