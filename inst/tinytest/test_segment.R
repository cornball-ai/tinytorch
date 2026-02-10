# test_segment.R - Tests for graph segmentation

# --- Single segment cases ---

# Pure graph expression -> 1 graph segment
segs1 <- segment_expr(quote({
  z <- x$matmul(y)
  z$relu()
}))
expect_equal(length(segs1), 1L)
expect_equal(segs1[[1]]$type, "graph")
expect_equal(length(segs1[[1]]$statements), 2L)

# Pure R code expression -> 1 r_code segment
segs2 <- segment_expr(quote({
  print(x)
  message("hello")
}))
expect_equal(length(segs2), 1L)
expect_equal(segs2[[1]]$type, "r_code")

# --- Multiple segment cases ---

# Graph, then break, then graph -> 3 segments
segs3 <- segment_expr(quote({
  z <- x$matmul(y)
  z <- z$relu()
  print(z$shape)
  result <- z$softmax(dim = 2L)
}))
expect_equal(length(segs3), 3L)
expect_equal(segs3[[1]]$type, "graph")
expect_equal(segs3[[2]]$type, "r_code")
expect_equal(segs3[[3]]$type, "graph")
expect_equal(length(segs3[[1]]$statements), 2L)  # matmul, relu
expect_equal(length(segs3[[2]]$statements), 1L)  # print
expect_equal(length(segs3[[3]]$statements), 1L)  # softmax

# Break at start
segs4 <- segment_expr(quote({
  print("starting")
  z <- x$matmul(y)
  z$relu()
}))
expect_equal(length(segs4), 2L)
expect_equal(segs4[[1]]$type, "r_code")
expect_equal(segs4[[2]]$type, "graph")

# Break at end
segs5 <- segment_expr(quote({
  z <- x$matmul(y)
  z <- z$relu()
  print(z)
}))
expect_equal(length(segs5), 2L)
expect_equal(segs5[[1]]$type, "graph")
expect_equal(segs5[[2]]$type, "r_code")

# Multiple breaks
segs6 <- segment_expr(quote({
  a <- x$matmul(y)
  print(a)
  b <- a$relu()
  message("done with relu")
  c <- b$softmax(dim = 2L)
}))
expect_equal(length(segs6), 5L)
types <- vapply(segs6, `[[`, character(1), "type")
expect_equal(types, c("graph", "r_code", "graph", "r_code", "graph"))

# --- Non-block expressions ---

# Single graph-safe expression
segs7 <- segment_expr(quote(x$matmul(y)))
expect_equal(length(segs7), 1L)
expect_equal(segs7[[1]]$type, "graph")

# Single graph-breaking expression
segs8 <- segment_expr(quote(print(x)))
expect_equal(length(segs8), 1L)
expect_equal(segs8[[1]]$type, "r_code")

# --- analyze_segments ---

analysis <- analyze_segments(quote({
  z <- x$matmul(y)
  print(z)
  z$relu()
}))
expect_equal(nrow(analysis), 3L)
expect_equal(analysis$type, c("graph", "r_code", "graph"))
expect_equal(analysis$n_statements, c(1L, 1L, 1L))

# --- Statements preserved correctly ---

# Check that original expressions are preserved
segs9 <- segment_expr(quote({
  z <- x$matmul(y)
  print(z)
}))
expect_equal(deparse(segs9[[1]]$statements[[1]]), "z <- x$matmul(y)")
expect_equal(deparse(segs9[[2]]$statements[[1]]), "print(z)")

# --- Complex graph expressions stay together ---

segs10 <- segment_expr(quote({
  # All of these should be in one graph segment
  a <- x$matmul(w1)
  a <- a$add(b1)
  a <- a$relu()
  b <- a$matmul(w2)
  b <- b$add(b2)
  b$softmax(dim = 2L)
}))
expect_equal(length(segs10), 1L)
expect_equal(segs10[[1]]$type, "graph")
expect_equal(length(segs10[[1]]$statements), 6L)
