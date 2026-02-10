# test_walk.R - Tests for AST walker

# --- Basic node classification ---

# Method call
node1 <- walk_expr(quote(x$matmul(y)))
expect_equal(node1$type, "method_call")
expect_equal(node1$method, "matmul")
expect_true(node1$is_torch_method)
expect_true(node1$graph_safe)

# Chained method call
node2 <- walk_expr(quote(x$matmul(y)$relu()))
expect_equal(node2$type, "method_call")
expect_equal(node2$method, "relu")
expect_equal(node2$object$type, "method_call")
expect_equal(node2$object$method, "matmul")

# torch_* function
node3 <- walk_expr(quote(torch_sum(x)))
expect_equal(node3$type, "torch_function")
expect_equal(node3$fn, "torch_sum")
expect_true(node3$graph_safe)

# Operators
node4 <- walk_expr(quote(x + y))
expect_equal(node4$type, "operator")
expect_equal(node4$op, "+")
expect_true(node4$graph_safe)

# Assignment
node5 <- walk_expr(quote(z <- x$matmul(y)))
expect_equal(node5$type, "assignment")
expect_equal(node5$target, "z")
expect_equal(node5$value$type, "method_call")

# Block expression
node6 <- walk_expr(quote({
  z <- x$matmul(y)
  z$relu()
}))
expect_equal(node6$type, "block")
expect_equal(node6$n_statements, 2L)
expect_equal(length(node6$statements), 2L)

# --- Graph safety ---

# Pure torch expressions are graph-safe
expect_true(is_graph_safe(quote(x$matmul(y))))
expect_true(is_graph_safe(quote(x$matmul(y)$relu())))
expect_true(is_graph_safe(quote(torch_sum(x))))
expect_true(is_graph_safe(quote(x + y * 2)))
expect_true(is_graph_safe(quote({
  z <- x$matmul(y)
  z$relu()
})))

# Side effects break the graph
expect_false(is_graph_safe(quote(print(x))))
expect_false(is_graph_safe(quote({
  z <- x$matmul(y)
  print(z)
  z$relu()
})))

# Control flow breaks the graph
expect_false(is_graph_safe(quote(if (cond) x$relu() else x$sigmoid())))
expect_false(is_graph_safe(quote(for (i in 1:10) x <- x$add(1))))

# Unknown functions break the graph (conservative)
expect_false(is_graph_safe(quote(my_custom_function(x))))

# --- Finding graph breaks ---

# No breaks in pure torch code
breaks1 <- find_graph_breaks(quote(x$matmul(y)$relu()))
expect_equal(length(breaks1), 0L)

# Print causes a break
breaks2 <- find_graph_breaks(quote({
  z <- x$matmul(y)
  print(z$shape)
  z$relu()
}))
expect_true(length(breaks2) > 0L)
expect_equal(breaks2[[1]]$type, "side_effect")
expect_equal(breaks2[[1]]$detail, "print")

# If statement causes a break
breaks3 <- find_graph_breaks(quote({
  if (cond) {
    x$relu()
  }
}))
expect_true(length(breaks3) > 0L)
expect_equal(breaks3[[1]]$type, "control_flow")
expect_equal(breaks3[[1]]$detail, "if")

# Unknown function causes a break
breaks4 <- find_graph_breaks(quote(my_function(x)))
expect_true(length(breaks4) > 0L)

# --- Nested structures ---

# Deeply nested method chains
node7 <- walk_expr(quote(x$matmul(y)$add(z)$relu()$softmax(dim = 2L)))
expect_equal(node7$type, "method_call")
expect_equal(node7$method, "softmax")
# Walk down the chain
obj <- node7$object
expect_equal(obj$method, "relu")
obj <- obj$object
expect_equal(obj$method, "add")
obj <- obj$object
expect_equal(obj$method, "matmul")

# Complex expression with multiple operations
node8 <- walk_expr(quote({
  a <- x$matmul(w1)
  b <- a$add(bias1)
  c <- b$relu()
  d <- c$matmul(w2)
  e <- d$add(bias2)
  e$softmax(dim = 2L)
}))
expect_equal(node8$type, "block")
expect_equal(node8$n_statements, 6L)
expect_true(is_graph_safe(node8))

# --- Symbols and literals ---

node_sym <- walk_expr(quote(x))
expect_equal(node_sym$type, "symbol")
expect_equal(node_sym$name, "x")

node_num <- walk_expr(quote(42))
expect_equal(node_num$type, "literal")
expect_equal(node_num$value, 42)

node_str <- walk_expr(quote("hello"))
expect_equal(node_str$type, "literal")
expect_equal(node_str$value, "hello")
