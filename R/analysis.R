#' Analysis Passes for IR Graphs
#'
#' Shape inference, liveness analysis, and dependency analysis.
#' These annotate IR graphs with information needed by optimization passes.

#' Broadcast Two Shapes (NumPy-compatible)
#'
#' Right-aligns dimensions and checks compatibility.
#' Each dimension pair must be equal, or one must be 1.
#'
#' @param shape_a Integer vector
#' @param shape_b Integer vector
#' @return Integer vector of broadcast shape, or NULL if incompatible
#' @export
broadcast_shapes <- function(shape_a, shape_b) {
  if (is.null(shape_a) || is.null(shape_b)) return(NULL)

  na <- length(shape_a)
  nb <- length(shape_b)
  n <- max(na, nb)

  # Right-align by padding with 1s on the left
  a <- c(rep(1L, n - na), as.integer(shape_a))
  b <- c(rep(1L, n - nb), as.integer(shape_b))

  result <- integer(n)
  for (i in seq_len(n)) {
    if (a[i] == b[i]) {
      result[i] <- a[i]
    } else if (a[i] == 1L) {
      result[i] <- b[i]
    } else if (b[i] == 1L) {
      result[i] <- a[i]
    } else {
      stop(sprintf("Incompatible shapes for broadcasting: [%s] vs [%s]",
                    paste(shape_a, collapse = ", "),
                    paste(shape_b, collapse = ", ")),
           call. = FALSE)
    }
  }
  result
}


#' Infer Shapes and Dtypes Through an IR Graph
#'
#' Forward pass: propagates shapes and dtypes from inputs through every node.
#' Returns a new ir_graph with \code{output_shape} and \code{output_dtype}
#' in each node's attrs.
#'
#' @param graph An ir_graph
#' @param input_shapes Named list mapping input names to integer vectors
#' @param input_dtypes Optional named list mapping input names to dtype strings
#' @return A new ir_graph with shape/dtype annotations
#' @export
infer_shapes <- function(graph, input_shapes, input_dtypes = NULL) {
  if (!inherits(graph, "ir_graph")) {
    stop("Expected an ir_graph object", call. = FALSE)
  }

  # Default dtypes
  if (is.null(input_dtypes)) {
    input_dtypes <- lapply(input_shapes, function(x) "float32")
  }

  # Clone nodes so we don't mutate the original
  nodes <- lapply(graph$nodes, function(n) {
    list(id = n$id, op = n$op, inputs = n$inputs,
         attrs = as.list(n$attrs))
  })
  names(nodes) <- names(graph$nodes)

  # Restore class on cloned nodes
  for (nm in names(nodes)) {
    class(nodes[[nm]]) <- "ir_node"
  }

  # Helper to get shape/dtype of a node by ID
  get_shape <- function(id) {
    nodes[[as.character(id)]]$attrs$output_shape
  }
  get_dtype <- function(id) {
    nodes[[as.character(id)]]$attrs$output_dtype
  }

  # Elementwise unary ops - same shape and dtype as input
  unary_ops <- c("relu", "sigmoid", "tanh", "exp", "log", "log2", "log10",
                 "sqrt", "abs", "neg", "sign", "floor", "ceil", "round",
                 "trunc", "gelu", "silu", "leaky_relu", "elu",
                 "softmax", "log_softmax", "contiguous", "clone", "detach",
                 "not")

  # Elementwise binary ops - broadcast shapes
  binary_ops <- c("add", "sub", "mul", "div", "pow", "remainder",
                  "floor_div", "and", "or")

  # Comparison ops - broadcast shapes, dtype becomes bool
  comparison_ops <- c("eq", "ne", "lt", "le", "gt", "ge")

  # Type cast ops - same shape, new dtype
  cast_ops <- list(
    float = "float32", double = "float64",
    int = "int32", long = "int64", bool = "bool"
  )

  # Reduction ops
  reduction_ops <- c("sum", "mean", "prod", "max", "min",
                     "torch_sum", "torch_mean", "torch_prod",
                     "torch_max", "torch_min")

  # Walk nodes in topological (ID) order
  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    op <- node$op
    inputs <- node$inputs
    attrs <- node$attrs

    shape <- NULL
    dtype <- NULL

    if (op == "input") {
      nm <- attrs$name
      if (!is.null(nm) && nm %in% names(input_shapes)) {
        shape <- as.integer(input_shapes[[nm]])
        dtype <- input_dtypes[[nm]] %||% "float32"
      }

    } else if (op == "constant") {
      val <- attrs$value
      if (is.numeric(val)) {
        shape <- as.integer(length(val))
        if (length(val) == 0L) shape <- integer(0)
        dtype <- if (is.integer(val)) "int32" else "float32"
      } else if (is.logical(val)) {
        shape <- as.integer(length(val))
        dtype <- "bool"
      }

    } else if (op %in% unary_ops) {
      if (length(inputs) >= 1L) {
        shape <- get_shape(inputs[1])
        dtype <- get_dtype(inputs[1])
      }

    } else if (op %in% binary_ops) {
      if (length(inputs) >= 2L) {
        s1 <- get_shape(inputs[1])
        s2 <- get_shape(inputs[2])
        shape <- tryCatch(broadcast_shapes(s1, s2), error = function(e) NULL)
        dtype <- get_dtype(inputs[1])
      }

    } else if (op %in% comparison_ops) {
      if (length(inputs) >= 2L) {
        s1 <- get_shape(inputs[1])
        s2 <- get_shape(inputs[2])
        shape <- tryCatch(broadcast_shapes(s1, s2), error = function(e) NULL)
        dtype <- "bool"
      }

    } else if (op == "matmul" || op == "mm") {
      if (length(inputs) >= 2L) {
        s1 <- get_shape(inputs[1])
        s2 <- get_shape(inputs[2])
        if (!is.null(s1) && !is.null(s2)) {
          n1 <- length(s1)
          n2 <- length(s2)
          if (n1 >= 2L && n2 >= 2L) {
            # [..., M, K] x [..., K, N] -> [..., M, N]
            m <- s1[n1 - 1L]
            k1 <- s1[n1]
            k2 <- s2[n2 - 1L]
            n <- s2[n2]
            if (k1 == k2) {
              # Broadcast batch dimensions
              if (n1 > 2L && n2 > 2L) {
                batch1 <- s1[seq_len(n1 - 2L)]
                batch2 <- s2[seq_len(n2 - 2L)]
                batch <- tryCatch(broadcast_shapes(batch1, batch2),
                                  error = function(e) NULL)
                if (!is.null(batch)) {
                  shape <- c(batch, m, n)
                }
              } else if (n1 > 2L) {
                shape <- c(s1[seq_len(n1 - 2L)], m, n)
              } else if (n2 > 2L) {
                shape <- c(s2[seq_len(n2 - 2L)], m, n)
              } else {
                shape <- c(m, n)
              }
            }
          } else if (n1 == 1L && n2 >= 2L) {
            # [K] x [..., K, N] -> [..., N]
            if (s1[1] == s2[n2 - 1L]) {
              shape <- s2[-c(n2 - 1L)]
              if (n2 == 2L) shape <- s2[n2]
            }
          } else if (n1 >= 2L && n2 == 1L) {
            # [..., M, K] x [K] -> [..., M]
            if (s1[n1] == s2[1]) {
              shape <- s1[-n1]
              if (n1 == 2L) shape <- s1[1]
            }
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "bmm") {
      if (length(inputs) >= 2L) {
        s1 <- get_shape(inputs[1])
        s2 <- get_shape(inputs[2])
        if (!is.null(s1) && !is.null(s2) &&
            length(s1) == 3L && length(s2) == 3L) {
          # [B, M, K] x [B, K, N] -> [B, M, N]
          if (s1[1] == s2[1] && s1[3] == s2[2]) {
            shape <- c(s1[1], s1[2], s2[3])
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op %in% reduction_ops) {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        if (!is.null(in_shape)) {
          dim_attr <- attrs$dim %||% attrs$arg1
          keepdim <- isTRUE(attrs$keepdim)

          if (is.null(dim_attr)) {
            # Full reduction -> scalar
            shape <- if (keepdim) rep(1L, length(in_shape)) else integer(0)
          } else {
            dim_idx <- as.integer(dim_attr)
            # R/torch uses 1-based dims
            if (dim_idx >= 1L && dim_idx <= length(in_shape)) {
              if (keepdim) {
                shape <- in_shape
                shape[dim_idx] <- 1L
              } else {
                shape <- in_shape[-dim_idx]
              }
            }
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "reshape" || op == "view") {
      if (length(inputs) >= 1L) {
        # Shape comes from attrs
        new_shape <- attrs$shape %||% attrs$arg1
        if (!is.null(new_shape)) {
          shape <- as.integer(new_shape)
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "squeeze") {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        if (!is.null(in_shape)) {
          dim_attr <- attrs$dim %||% attrs$arg1
          if (is.null(dim_attr)) {
            # Remove all dims of size 1
            shape <- in_shape[in_shape != 1L]
            if (length(shape) == 0L) shape <- 1L
          } else {
            dim_idx <- as.integer(dim_attr)
            if (dim_idx >= 1L && dim_idx <= length(in_shape) &&
                in_shape[dim_idx] == 1L) {
              shape <- in_shape[-dim_idx]
              if (length(shape) == 0L) shape <- 1L
            } else {
              shape <- in_shape
            }
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "unsqueeze") {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        dim_attr <- attrs$dim %||% attrs$arg1
        if (!is.null(in_shape) && !is.null(dim_attr)) {
          dim_idx <- as.integer(dim_attr)
          # Insert 1 at the specified position
          if (dim_idx >= 1L && dim_idx <= length(in_shape) + 1L) {
            shape <- append(in_shape, 1L, after = dim_idx - 1L)
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "transpose") {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        if (!is.null(in_shape)) {
          dim0 <- attrs$dim0 %||% attrs$arg1
          dim1 <- attrs$dim1 %||% attrs$arg2
          if (!is.null(dim0) && !is.null(dim1)) {
            d0 <- as.integer(dim0)
            d1 <- as.integer(dim1)
            if (d0 >= 1L && d0 <= length(in_shape) &&
                d1 >= 1L && d1 <= length(in_shape)) {
              shape <- in_shape
              shape[d0] <- in_shape[d1]
              shape[d1] <- in_shape[d0]
            }
          } else if (length(in_shape) == 2L) {
            # Default: swap last two dims
            shape <- rev(in_shape)
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "permute") {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        perm <- attrs$dims %||% attrs$arg1
        if (!is.null(in_shape) && !is.null(perm)) {
          perm <- as.integer(perm)
          shape <- in_shape[perm]
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "flatten") {
      if (length(inputs) >= 1L) {
        in_shape <- get_shape(inputs[1])
        if (!is.null(in_shape)) {
          start_dim <- as.integer(attrs$start_dim %||% attrs$arg1 %||% 1L)
          end_dim <- as.integer(attrs$end_dim %||% attrs$arg2 %||% length(in_shape))
          if (end_dim < 0L) end_dim <- length(in_shape) + end_dim + 1L
          if (start_dim >= 1L && end_dim <= length(in_shape) &&
              start_dim <= end_dim) {
            flat_size <- prod(in_shape[start_dim:end_dim])
            before <- if (start_dim > 1L) in_shape[seq_len(start_dim - 1L)] else integer(0)
            after <- if (end_dim < length(in_shape)) in_shape[(end_dim + 1L):length(in_shape)] else integer(0)
            shape <- c(before, as.integer(flat_size), after)
          }
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "torch_linear") {
      # torch_linear(input, weight, bias): input [..., K] x weight [N, K] -> [..., N]
      if (length(inputs) >= 2L) {
        s_input <- get_shape(inputs[1])
        s_weight <- get_shape(inputs[2])
        if (!is.null(s_input) && !is.null(s_weight) &&
            length(s_input) >= 1L && length(s_weight) == 2L) {
          # Replace last dim with weight's first dim (output features)
          shape <- s_input
          shape[length(shape)] <- s_weight[1]
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "torch_layer_norm") {
      # layer_norm: same shape as input
      if (length(inputs) >= 1L) {
        shape <- get_shape(inputs[1])
        dtype <- get_dtype(inputs[1])
      }

    } else if (op %in% c("torch_gelu", "torch_silu", "torch_sigmoid")) {
      # Elementwise: same shape and dtype
      if (length(inputs) >= 1L) {
        shape <- get_shape(inputs[1])
        dtype <- get_dtype(inputs[1])
      }

    } else if (op == "torch_conv1d") {
      # conv1d(input [N,C_in,L], weight [C_out,C_in/groups,K]) -> [N,C_out,L_out]
      if (length(inputs) >= 2L) {
        s_input <- get_shape(inputs[1])
        s_weight <- get_shape(inputs[2])
        if (!is.null(s_input) && !is.null(s_weight) &&
            length(s_input) == 3L && length(s_weight) == 3L) {
          stride <- as.integer(attrs$stride %||% 1L)
          padding <- as.integer(attrs$padding %||% 0L)
          dilation <- as.integer(attrs$dilation %||% 1L)
          L_in <- s_input[3]
          K_size <- s_weight[3]
          L_out <- as.integer(floor((L_in + 2L * padding - dilation * (K_size - 1L) - 1L) / stride + 1L))
          shape <- c(s_input[1], s_weight[1], L_out)
        }
        dtype <- get_dtype(inputs[1])
      }

    } else if (op %in% names(cast_ops)) {
      if (length(inputs) >= 1L) {
        shape <- get_shape(inputs[1])
        dtype <- cast_ops[[op]]
      }

    } else {
      # Unknown op: try to propagate from first input
      if (length(inputs) >= 1L) {
        shape <- get_shape(inputs[1])
        dtype <- get_dtype(inputs[1])
      }
    }

    nodes[[id_str]]$attrs$output_shape <- shape
    nodes[[id_str]]$attrs$output_dtype <- dtype
  }

  ir_graph(nodes, graph$input_ids, graph$output_ids)
}


#' Liveness Analysis
#'
#' Backward pass: for each node, compute the last point its output is consumed.
#' Nodes in \code{output_ids} have \code{last_use = Inf} (live at graph exit).
#' Nodes with no consumers get \code{last_use = NA} (dead code).
#'
#' @param graph An ir_graph
#' @return Named list of node_id (as character) to last_use_id (integer or Inf)
#' @export
analyze_liveness <- function(graph) {
  if (!inherits(graph, "ir_graph")) {
    stop("Expected an ir_graph object", call. = FALSE)
  }

  ids <- sort(as.integer(names(graph$nodes)))
  last_use <- rep(NA_real_, length(ids))
  names(last_use) <- as.character(ids)

  # Output nodes live to infinity
  for (out_id in graph$output_ids) {
    last_use[as.character(out_id)] <- Inf
  }

  # Walk in reverse topological order
  for (id in rev(ids)) {
    node <- graph$nodes[[as.character(id)]]
    for (inp_id in node$inputs) {
      inp_str <- as.character(inp_id)
      current <- last_use[inp_str]
      if (is.na(current) || id > current) {
        last_use[inp_str] <- id
      }
    }
  }

  as.list(last_use)
}


#' Dependency Analysis
#'
#' For each node, compute the full transitive dependency set.
#'
#' @param graph An ir_graph
#' @return Named list of node_id (as character) to integer vector of dependency IDs
#' @export
analyze_deps <- function(graph) {
  if (!inherits(graph, "ir_graph")) {
    stop("Expected an ir_graph object", call. = FALSE)
  }

  ids <- sort(as.integer(names(graph$nodes)))
  deps <- list()

  for (id in ids) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]

    if (length(node$inputs) == 0L) {
      # Input/constant: no dependencies
      deps[[id_str]] <- integer(0)
    } else {
      # Union of all input deps + input IDs themselves
      all_deps <- integer(0)
      for (inp_id in node$inputs) {
        inp_str <- as.character(inp_id)
        all_deps <- c(all_deps, inp_id, deps[[inp_str]])
      }
      deps[[id_str]] <- sort(unique(all_deps))
    }
  }

  deps
}
