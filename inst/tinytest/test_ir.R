# test_ir.R - Tests for IR foundation

# --- IR node construction ---

n1 <- Rtorch:::ir_node(1L, "input", attrs = list(name = "x"))
expect_equal(n1$id, 1L)
expect_equal(n1$op, "input")
expect_equal(n1$attrs$name, "x")
expect_equal(length(n1$inputs), 0L)

n2 <- Rtorch:::ir_node(2L, "relu", inputs = 1L)
expect_equal(n2$inputs, 1L)

# --- Simple chain: x$relu()$sigmoid() ---

ir1 <- lower_to_ir(list(quote(x$relu()$sigmoid())))
expect_true(validate_ir(ir1))
expect_equal(length(ir1$nodes), 3L)  # input, relu, sigmoid
expect_equal(length(ir1$input_ids), 1L)

# Check structure: input -> relu -> sigmoid
nodes1 <- ir1$nodes
input_node <- nodes1[[as.character(ir1$input_ids[1])]]
expect_equal(input_node$op, "input")
expect_equal(input_node$attrs$name, "x")

# The output should be the sigmoid node
out_node <- nodes1[[as.character(ir1$output_ids[1])]]
expect_equal(out_node$op, "sigmoid")

# --- Binary operations: x + y ---

ir2 <- lower_to_ir(list(quote(x + y)))
expect_true(validate_ir(ir2))
expect_equal(length(ir2$input_ids), 2L)  # x and y

out2 <- ir2$nodes[[as.character(ir2$output_ids[1])]]
expect_equal(out2$op, "add")
expect_equal(length(out2$inputs), 2L)

# --- Method call with args: x$matmul(y) ---

ir3 <- lower_to_ir(list(quote(x$matmul(y))))
expect_true(validate_ir(ir3))
expect_equal(length(ir3$input_ids), 2L)

out3 <- ir3$nodes[[as.character(ir3$output_ids[1])]]
expect_equal(out3$op, "matmul")
expect_equal(length(out3$inputs), 2L)

# --- Constants and literals ---

ir4 <- lower_to_ir(list(quote(x + 1)))
expect_true(validate_ir(ir4))

# Should have: input[x], constant(1), add
expect_equal(length(ir4$nodes), 3L)
expect_equal(length(ir4$input_ids), 1L)  # only x is an input

# Find the constant node
const_nodes <- Filter(function(n) n$op == "constant", ir4$nodes)
expect_equal(length(const_nodes), 1L)
expect_equal(const_nodes[[1]]$attrs$value, 1)

# --- Assignment: z <- x$relu() ---

ir5 <- lower_to_ir(list(quote(z <- x$relu())))
expect_true(validate_ir(ir5))

# Assignment doesn't create a node - z maps to relu's output
# So we should have: input[x], relu
expect_equal(length(ir5$nodes), 2L)

# --- Multi-statement blocks ---

ir6 <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
expect_true(validate_ir(ir6))

# input[x], input[y], matmul, relu
expect_equal(length(ir6$nodes), 4L)
expect_equal(length(ir6$input_ids), 2L)

# z in second statement should reference matmul's output
out6 <- ir6$nodes[[as.character(ir6$output_ids[1])]]
expect_equal(out6$op, "relu")
# relu's input should be matmul's ID
matmul_node <- Filter(function(n) n$op == "matmul", ir6$nodes)[[1]]
expect_equal(out6$inputs, matmul_node$id)

# --- Longer chain with assignment ---

ir7 <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(w <- z$relu()),
  quote(w$sigmoid())
))
expect_true(validate_ir(ir7))

# input[x], input[y], matmul, relu, sigmoid
expect_equal(length(ir7$nodes), 5L)
out7 <- ir7$nodes[[as.character(ir7$output_ids[1])]]
expect_equal(out7$op, "sigmoid")

# --- Operators: x * y + z ---

ir8 <- lower_to_ir(list(quote(x * y + z)))
expect_true(validate_ir(ir8))
expect_equal(length(ir8$input_ids), 3L)

out8 <- ir8$nodes[[as.character(ir8$output_ids[1])]]
expect_equal(out8$op, "add")

# --- Unary negation ---

ir9 <- lower_to_ir(list(quote(-x)))
expect_true(validate_ir(ir9))
out9 <- ir9$nodes[[as.character(ir9$output_ids[1])]]
expect_equal(out9$op, "neg")

# --- torch_* functions ---

ir10 <- lower_to_ir(list(quote(torch_sum(x))))
expect_true(validate_ir(ir10))
out10 <- ir10$nodes[[as.character(ir10$output_ids[1])]]
expect_equal(out10$op, "torch_sum")
expect_equal(out10$attrs$fn, "torch_sum")

# --- Block expression lowering ---

ir11 <- lower_to_ir(list(quote({
  z <- x$matmul(y)
  z$relu()$sigmoid()
})))
expect_true(validate_ir(ir11))

# input[x], input[y], matmul, relu, sigmoid
expect_equal(length(ir11$nodes), 5L)
out11 <- ir11$nodes[[as.character(ir11$output_ids[1])]]
expect_equal(out11$op, "sigmoid")

# --- Variable reuse across statements ---

ir12 <- lower_to_ir(list(
  quote(a <- x$relu()),
  quote(b <- x$sigmoid()),
  quote(a + b)
))
expect_true(validate_ir(ir12))

# input[x], relu, sigmoid, add -- x is shared
expect_equal(length(ir12$input_ids), 1L)  # x only created once
expect_equal(length(ir12$nodes), 4L)

# --- Printer output ---

ir_print <- lower_to_ir(list(
  quote(z <- x$matmul(y)),
  quote(z$relu())
))
output <- capture.output(print(ir_print))
expect_true(any(grepl("input\\[x\\]", output)))
expect_true(any(grepl("input\\[y\\]", output)))
expect_true(any(grepl("matmul", output)))
expect_true(any(grepl("relu", output)))
expect_true(any(grepl("return %", output)))

# --- Validator catches bad graphs ---

# Bad reference
bad_graph <- Rtorch:::ir_graph(
  nodes = list("1" = Rtorch:::ir_node(1L, "relu", inputs = 99L)),
  input_ids = integer(),
  output_ids = 1L
)
expect_error(validate_ir(bad_graph), "non-existent input")

# Bad output reference
bad_graph2 <- Rtorch:::ir_graph(
  nodes = list("1" = Rtorch:::ir_node(1L, "input")),
  input_ids = 1L,
  output_ids = 99L
)
expect_error(validate_ir(bad_graph2), "non-existent node")

# Not an ir_graph
expect_error(validate_ir(list()), "ir_graph")

# --- Topological order check ---

bad_topo <- Rtorch:::ir_graph(
  nodes = list(
    "2" = Rtorch:::ir_node(2L, "relu", inputs = 3L),
    "3" = Rtorch:::ir_node(3L, "input")
  ),
  input_ids = 3L,
  output_ids = 2L
)
expect_error(validate_ir(bad_topo), "topological order")

# --- Comparison operators ---

ir_cmp <- lower_to_ir(list(quote(x > y)))
expect_true(validate_ir(ir_cmp))
out_cmp <- ir_cmp$nodes[[as.character(ir_cmp$output_ids[1])]]
expect_equal(out_cmp$op, "gt")

# --- format_ir_node ---

n_input <- Rtorch:::ir_node(1L, "input", attrs = list(name = "x"))
expect_true(grepl("input\\[x\\]", Rtorch:::format_ir_node(n_input)))

n_const <- Rtorch:::ir_node(2L, "constant", attrs = list(value = 42))
expect_true(grepl("constant\\(42\\)", Rtorch:::format_ir_node(n_const)))

n_relu <- Rtorch:::ir_node(3L, "relu", inputs = 1L)
expect_true(grepl("relu\\(%1\\)", Rtorch:::format_ir_node(n_relu)))

n_add <- Rtorch:::ir_node(4L, "add", inputs = c(1L, 2L))
expect_true(grepl("add\\(%1, %2\\)", Rtorch:::format_ir_node(n_add)))
