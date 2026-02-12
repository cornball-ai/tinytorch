#' Optimization Passes for IR Graphs
#'
#' Each pass takes an ir_graph and returns a rewritten ir_graph.
#' Passes are composable via optimize_graph().

# --- Internal helpers ---

# Evaluate a constant operation
.eval_constant_op <- function(op, vals) {
  tryCatch(switch(op,
    add = vals[[1]] + vals[[2]],
    sub = vals[[1]] - vals[[2]],
    mul = vals[[1]] * vals[[2]],
    div = vals[[1]] / vals[[2]],
    pow = vals[[1]] ^ vals[[2]],
    remainder = vals[[1]] %% vals[[2]],
    floor_div = vals[[1]] %/% vals[[2]],
    neg = -vals[[1]],
    abs = abs(vals[[1]]),
    sqrt = sqrt(vals[[1]]),
    rsqrt = 1 / sqrt(vals[[1]]),
    exp = exp(vals[[1]]),
    log = log(vals[[1]]),
    log2 = log2(vals[[1]]),
    log10 = log10(vals[[1]]),
    sin = sin(vals[[1]]),
    cos = cos(vals[[1]]),
    floor = floor(vals[[1]]),
    ceil = ceiling(vals[[1]]),
    round = round(vals[[1]]),
    trunc = trunc(vals[[1]]),
    sign = sign(vals[[1]]),
    NULL
  ), error = function(e) NULL)
}

# Check if a node is a scalar constant with a specific value
.is_scalar_const <- function(nodes, node_id, value) {
  n <- nodes[[as.character(node_id)]]
  if (is.null(n) || n$op != "constant") return(FALSE)
  v <- n$attrs$value
  is.numeric(v) && length(v) == 1L && isTRUE(all.equal(v, value))
}

# Follow redirect chains to resolve final target
.resolve <- function(id, redirect) {
  id_str <- as.character(id)
  seen <- character()
  while (id_str %in% names(redirect)) {
    if (id_str %in% seen) break
    seen <- c(seen, id_str)
    id <- redirect[[id_str]]
    id_str <- as.character(id)
  }
  as.integer(id)
}

# Apply redirect map to all node inputs and output_ids
.apply_redirects <- function(nodes, output_ids, redirect) {
  if (length(redirect) == 0L) {
    return(list(nodes = nodes, output_ids = output_ids))
  }
  for (id_str in names(nodes)) {
    inps <- nodes[[id_str]]$inputs
    if (length(inps) > 0L) {
      nodes[[id_str]]$inputs <- as.integer(vapply(
        inps, .resolve, integer(1), redirect = redirect
      ))
    }
  }
  output_ids <- as.integer(vapply(
    output_ids, .resolve, integer(1), redirect = redirect
  ))
  list(nodes = nodes, output_ids = output_ids)
}

# Hash a node for CSE (excludes analysis-only attrs)
.node_cse_hash <- function(op, inputs, attrs) {
  semantic_attrs <- attrs[!names(attrs) %in%
    c("output_shape", "output_dtype", "fusion_group")]
  attr_str <- if (length(semantic_attrs) > 0) {
    paste(deparse(semantic_attrs, control = "all"), collapse = "")
  } else {
    ""
  }
  paste(op, paste(inputs, collapse = ","), attr_str, sep = "|")
}

# Deep-clone nodes list
.clone_nodes <- function(nodes) {
  out <- lapply(nodes, function(n) {
    ir_node(n$id, n$op, n$inputs, as.list(n$attrs))
  })
  names(out) <- names(nodes)
  out
}


#' Constant Folding
#'
#' If all inputs to a node are constants, evaluate at compile time
#' and replace the node with a constant.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with constants folded
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' constant_fold(g)
#' }
#' @export
constant_fold <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)
  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    if (node$op %in% c("constant", "input") || length(node$inputs) == 0L) next

    all_const <- all(vapply(node$inputs, function(inp_id) {
      nodes[[as.character(inp_id)]]$op == "constant"
    }, logical(1)))
    if (!all_const) next

    input_vals <- lapply(node$inputs, function(inp_id) {
      nodes[[as.character(inp_id)]]$attrs$value
    })

    result <- .eval_constant_op(node$op, input_vals)
    if (is.null(result)) next

    nodes[[id_str]] <- ir_node(id, "constant", integer(0), list(value = result))
  }

  ir_graph(nodes, graph$input_ids, graph$output_ids)
}


#' Dead Code Elimination
#'
#' Remove nodes whose outputs are never consumed.
#' Walks backward from outputs to find reachable nodes.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with dead nodes removed
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' dead_code_eliminate(g)
#' }
#' @export
dead_code_eliminate <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  reachable <- integer()

  mark <- function(node_id) {
    if (node_id %in% reachable) return()
    reachable <<- c(reachable, node_id)
    node <- graph$nodes[[as.character(node_id)]]
    if (!is.null(node)) {
      for (inp_id in node$inputs) mark(inp_id)
    }
  }

  for (out_id in graph$output_ids) mark(out_id)

  reachable_strs <- as.character(sort(reachable))
  nodes <- graph$nodes[reachable_strs]
  input_ids <- graph$input_ids[graph$input_ids %in% reachable]

  ir_graph(nodes, input_ids, graph$output_ids)
}


#' Common Subexpression Elimination
#'
#' If two nodes have the same op, same inputs, and same attrs,
#' replace the second with a reference to the first.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with duplicate expressions eliminated
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' common_subexpr_eliminate(g)
#' }
#' @export
common_subexpr_eliminate <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)
  seen <- list()
  redirect <- list()

  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    if (node$op == "input") next

    # Resolve inputs through existing redirects before hashing
    resolved_inputs <- if (length(node$inputs) > 0L) {
      as.integer(vapply(node$inputs, .resolve, integer(1), redirect = redirect))
    } else {
      node$inputs
    }

    h <- .node_cse_hash(node$op, resolved_inputs, node$attrs)

    if (!is.null(seen[[h]])) {
      redirect[[id_str]] <- seen[[h]]
    } else {
      seen[[h]] <- node$id
      nodes[[id_str]]$inputs <- resolved_inputs
    }
  }

  result <- .apply_redirects(nodes, graph$output_ids, redirect)

  # Remove redirected nodes
  for (old_str in names(redirect)) {
    result$nodes[[old_str]] <- NULL
  }

  remaining_ids <- as.integer(names(result$nodes))
  input_ids <- graph$input_ids[graph$input_ids %in% remaining_ids]

  ir_graph(result$nodes, input_ids, result$output_ids)
}


#' Algebraic Simplification
#'
#' Pattern-match and rewrite known algebraic identities.
#' Assumes fast-math semantics (no NaN/Inf guards).
#'
#' @param graph An ir_graph
#' @return A new ir_graph with algebraic simplifications applied
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$mul(torch_tensor(1.0))))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' algebraic_simplify(g)
#' }
#' @export
algebraic_simplify <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)
  redirect <- list()

  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    op <- node$op
    inputs <- node$inputs

    # --- Addition identities ---
    if (op == "add" && length(inputs) == 2L) {
      if (.is_scalar_const(nodes, inputs[2], 0)) {
        redirect[[id_str]] <- inputs[1]; next
      }
      if (.is_scalar_const(nodes, inputs[1], 0)) {
        redirect[[id_str]] <- inputs[2]; next
      }
    }

    # --- Multiplication identities ---
    if (op == "mul" && length(inputs) == 2L) {
      # x * 1 -> x
      if (.is_scalar_const(nodes, inputs[2], 1)) {
        redirect[[id_str]] <- inputs[1]; next
      }
      if (.is_scalar_const(nodes, inputs[1], 1)) {
        redirect[[id_str]] <- inputs[2]; next
      }
      # x * 0 -> 0
      if (.is_scalar_const(nodes, inputs[1], 0) ||
          .is_scalar_const(nodes, inputs[2], 0)) {
        nodes[[id_str]] <- ir_node(id, "constant", integer(0), list(value = 0))
        next
      }
      # sigmoid(x) * x -> silu(x)
      n1 <- nodes[[as.character(inputs[1])]]
      n2 <- nodes[[as.character(inputs[2])]]
      if (!is.null(n1) && n1$op == "sigmoid" && length(n1$inputs) == 1L &&
          n1$inputs[1] == inputs[2]) {
        nodes[[id_str]] <- ir_node(id, "silu", inputs[2]); next
      }
      if (!is.null(n2) && n2$op == "sigmoid" && length(n2$inputs) == 1L &&
          n2$inputs[1] == inputs[1]) {
        nodes[[id_str]] <- ir_node(id, "silu", inputs[1]); next
      }
    }

    # --- Self-cancellation ---
    if (op == "sub" && length(inputs) == 2L && inputs[1] == inputs[2]) {
      nodes[[id_str]] <- ir_node(id, "constant", integer(0), list(value = 0))
      next
    }
    if (op == "div" && length(inputs) == 2L && inputs[1] == inputs[2]) {
      nodes[[id_str]] <- ir_node(id, "constant", integer(0), list(value = 1))
      next
    }

    # --- Power simplifications ---
    if (op == "pow" && length(inputs) == 2L) {
      if (.is_scalar_const(nodes, inputs[2], 2)) {
        nodes[[id_str]] <- ir_node(id, "mul", c(inputs[1], inputs[1]))
        next
      }
      if (.is_scalar_const(nodes, inputs[2], 0.5)) {
        nodes[[id_str]] <- ir_node(id, "sqrt", inputs[1])
        next
      }
    }

    # --- Double negation ---
    if (op == "neg" && length(inputs) == 1L) {
      inner <- nodes[[as.character(inputs[1])]]
      if (!is.null(inner) && inner$op == "neg" && length(inner$inputs) == 1L) {
        redirect[[id_str]] <- inner$inputs[1]; next
      }
    }

    # --- Idempotent relu ---
    if (op == "relu" && length(inputs) == 1L) {
      inner <- nodes[[as.character(inputs[1])]]
      if (!is.null(inner) && inner$op == "relu") {
        redirect[[id_str]] <- inputs[1]; next
      }
    }
  }

  result <- .apply_redirects(nodes, graph$output_ids, redirect)
  ir_graph(result$nodes, graph$input_ids, result$output_ids)
}


#' Decompose High-Level Ops to Primitives
#'
#' Rewrites composite ops into lower-level primitives to expose
#' fuseable elementwise chains:
#' \itemize{
#'   \item \code{torch_linear(x, w, b)} -> \code{transpose(w) + matmul(x, wt) + add(result, b)}
#'   \item \code{torch_gelu(x)} -> \code{gelu(x)}
#'   \item \code{torch_silu(x)} -> \code{silu(x)}
#'   \item \code{torch_sigmoid(x)} -> \code{sigmoid(x)}
#' }
#' \code{torch_layer_norm} and \code{torch_conv1d} are kept as-is
#' (they contain reductions or special layouts that can't be
#' elementwise-fused).
#'
#' @param graph An ir_graph
#' @return A new ir_graph with decomposed ops
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' decompose_high_level_ops(g)
#' }
#' @export
decompose_high_level_ops <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)
  output_ids <- graph$output_ids
  next_id <- max(as.integer(names(nodes))) + 1L

  new_id <- function() {
    id <- next_id
    next_id <<- next_id + 1L
    id
  }

  add_new_node <- function(id, op, inputs, attrs = list()) {
    node <- ir_node(id, op, inputs, attrs)
    nodes[[as.character(id)]] <<- node
    id
  }

  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    op <- node$op

    if (op == "torch_linear") {
      # Keep torch_linear as-is: executor handles it natively via
      # cuBLAS (GPU) or MKL (CPU), which is faster and more numerically
      # stable than decomposing to transpose + matmul + add.

    } else if (op == "torch_gelu") {
      # torch_gelu(x) -> gelu(x)
      nodes[[id_str]] <- ir_node(id, "gelu", node$inputs)

    } else if (op == "torch_silu") {
      # torch_silu(x) -> silu(x)
      nodes[[id_str]] <- ir_node(id, "silu", node$inputs)

    } else if (op == "torch_sigmoid") {
      # torch_sigmoid(x) -> sigmoid(x)
      nodes[[id_str]] <- ir_node(id, "sigmoid", node$inputs)
    }
    # torch_layer_norm, torch_conv1d: keep as-is
  }

  ir_graph(nodes, graph$input_ids, output_ids)
}


#' Fusion Annotation
#'
#' Identifies maximal fuseable subgraphs: contiguous chains of
#' elementwise operations where each intermediate result has
#' exactly one consumer. Annotates nodes with a \code{fusion_group} ID.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with fusion_group annotations
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()$sigmoid()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' fusion_annotate(g)
#' }
#' @export
fusion_annotate <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  elementwise_ops <- c(
    "relu", "sigmoid", "tanh", "exp", "log", "log2", "log10",
    "sqrt", "rsqrt", "abs", "neg", "sign", "floor", "ceil", "round", "trunc",
    "sin", "cos",
    "gelu", "silu", "leaky_relu", "elu",
    "add", "sub", "mul", "div", "pow", "remainder", "floor_div"
  )

  # Build consumer map
  consumers <- list()
  for (node in graph$nodes) {
    for (inp_id in node$inputs) {
      key <- as.character(inp_id)
      consumers[[key]] <- c(consumers[[key]], node$id)
    }
  }

  nodes <- .clone_nodes(graph$nodes)
  next_group <- 1L
  node_group <- list()

  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]
    if (!node$op %in% elementwise_ops) next

    # Try to inherit fusion group from a sole-consumer elementwise input
    inherited <- NULL
    for (inp_id in node$inputs) {
      inp_str <- as.character(inp_id)
      inp_node <- nodes[[inp_str]]
      if (is.null(inp_node)) next
      if (inp_node$op %in% elementwise_ops &&
          length(consumers[[inp_str]]) == 1L &&
          !is.null(node_group[[inp_str]])) {
        inherited <- node_group[[inp_str]]
        break
      }
    }

    if (!is.null(inherited)) {
      node_group[[id_str]] <- inherited
    } else {
      node_group[[id_str]] <- next_group
      next_group <- next_group + 1L
    }
  }

  # Only keep groups with >= 2 members
  if (length(node_group) > 0L) {
    group_ids <- unlist(node_group)
    group_counts <- table(group_ids)
    valid_groups <- as.integer(names(group_counts[group_counts >= 2L]))

    for (id_str in names(node_group)) {
      if (node_group[[id_str]] %in% valid_groups) {
        nodes[[id_str]]$attrs$fusion_group <- as.integer(node_group[[id_str]])
      }
    }
  }

  ir_graph(nodes, graph$input_ids, graph$output_ids)
}


#' Detect Matmul Epilogue Fusion Patterns
#'
#' Finds fusion groups where the entry node is \code{add(matmul_result, bias)}
#' and the matmul is not in any fusion group. The remaining ops in the group
#' must all be unary (gelu, relu, silu, etc.). Returns pattern info for
#' each match, suitable for compilation into fused matmul+bias+epilogue
#' GPU kernels.
#'
#' @param graph An ir_graph with fusion_group annotations
#' @return List of pattern descriptors, each with matmul_id, bias_id,
#'   epilogue_ops, group_id, all_node_ids, output_id, and the matmul's
#'   A and B input IDs. Empty list if no patterns found.
#' @noRd
detect_matmul_epilogues <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  unary_ops <- c(
    "relu", "sigmoid", "tanh", "exp", "log", "log2", "sqrt", "rsqrt",
    "abs", "neg", "sign", "sin", "cos",
    "silu", "gelu", "leaky_relu", "elu"
  )

  # Build map: group_id -> sorted node IDs
  group_map <- list()
  for (id_str in names(graph$nodes)) {
    gid <- graph$nodes[[id_str]]$attrs$fusion_group
    if (!is.null(gid)) {
      gid_str <- as.character(gid)
      group_map[[gid_str]] <- c(group_map[[gid_str]], as.integer(id_str))
    }
  }
  for (gid_str in names(group_map)) {
    group_map[[gid_str]] <- sort(group_map[[gid_str]])
  }

  # Set of all nodes in any fusion group (for checking matmul isn't in one)
  all_fused <- character()
  for (nids in group_map) all_fused <- c(all_fused, as.character(nids))

  patterns <- list()

  for (gid_str in names(group_map)) {
    nids <- group_map[[gid_str]]
    entry_id <- min(nids)
    entry_node <- graph$nodes[[as.character(entry_id)]]

    # Entry must be an "add" (bias add)
    if (entry_node$op != "add" || length(entry_node$inputs) != 2L) next

    # One input must be a matmul NOT in any fusion group
    matmul_id <- NULL
    bias_id <- NULL
    for (inp in entry_node$inputs) {
      inp_str <- as.character(inp)
      inp_node <- graph$nodes[[inp_str]]
      if (is.null(inp_node)) next
      if (inp_node$op == "matmul" && !inp_str %in% all_fused) {
        matmul_id <- inp
      } else {
        bias_id <- inp
      }
    }
    if (is.null(matmul_id) || is.null(bias_id)) next

    # All remaining group nodes (after entry) must be unary
    rest_ids <- nids[nids != entry_id]
    all_unary <- TRUE
    epilogue_ops <- character()
    for (rid in sort(rest_ids)) {
      rnode <- graph$nodes[[as.character(rid)]]
      if (!rnode$op %in% unary_ops) {
        all_unary <- FALSE
        break
      }
      epilogue_ops <- c(epilogue_ops, rnode$op)
    }
    if (!all_unary) next

    # Get matmul's inputs (A, B)
    mm_node <- graph$nodes[[as.character(matmul_id)]]

    patterns[[length(patterns) + 1L]] <- list(
      matmul_id = matmul_id,
      a_input_id = mm_node$inputs[1],
      b_input_id = mm_node$inputs[2],
      bias_id = bias_id,
      epilogue_ops = epilogue_ops,
      group_id = as.integer(gid_str),
      all_node_ids = nids,
      output_id = max(nids)
    )
  }

  patterns
}


#' Detect Scaled Dot-Product Attention Patterns
#'
#' Finds the manual attention pattern in the IR graph:
#' \code{matmul(Q, K^T) -> mul(_, scale) -> [add(_, mask)] -> softmax(_, -1) -> matmul(_, V)}
#' and rewrites it to a single \code{torch_sdpa} node that dispatches to
#' FlashAttention on GPU.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with SDPA patterns fused
#' @noRd
detect_sdpa_patterns <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)
  redirect <- list()

  # Build consumer map: node_id -> list of consumer node_ids
  consumers <- list()
  for (node in nodes) {
    for (inp_id in node$inputs) {
      key <- as.character(inp_id)
      consumers[[key]] <- c(consumers[[key]], node$id)
    }
  }

  # Helper: check if a node has exactly one consumer
  single_consumer <- function(id) {
    length(consumers[[as.character(id)]]) == 1L
  }

  ids <- sort(as.integer(names(nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- nodes[[id_str]]

    # Look for the final matmul: matmul(softmax_output, V)
    if (node$op != "matmul" || length(node$inputs) != 2L) next

    softmax_id <- node$inputs[1]
    v_id <- node$inputs[2]
    softmax_node <- nodes[[as.character(softmax_id)]]

    # The first input must be softmax with dim=-1
    if (is.null(softmax_node) || softmax_node$op != "softmax") next
    sdim <- softmax_node$attrs$dim %||% softmax_node$attrs$arg1
    if (!is.null(sdim) && sdim != -1L) next
    if (!single_consumer(softmax_id)) next

    # softmax input: either add(scores, mask) or mul(matmul_result, scale)
    softmax_inp_id <- softmax_node$inputs[1]
    softmax_inp <- nodes[[as.character(softmax_inp_id)]]
    if (is.null(softmax_inp)) next

    mask_id <- NULL
    scores_id <- NULL

    if (softmax_inp$op == "add" && length(softmax_inp$inputs) == 2L &&
        single_consumer(softmax_inp_id)) {
      # Pattern with mask: add(scaled_scores, mask)
      # Figure out which input is the scaled scores vs the mask
      # The scaled scores come from mul(matmul_result, scale)
      inp1_node <- nodes[[as.character(softmax_inp$inputs[1])]]
      inp2_node <- nodes[[as.character(softmax_inp$inputs[2])]]

      if (!is.null(inp1_node) && inp1_node$op == "mul") {
        scores_id <- softmax_inp$inputs[1]
        mask_id <- softmax_inp$inputs[2]
      } else if (!is.null(inp2_node) && inp2_node$op == "mul") {
        scores_id <- softmax_inp$inputs[2]
        mask_id <- softmax_inp$inputs[1]
      } else {
        next
      }
    } else if (softmax_inp$op == "mul" && single_consumer(softmax_inp_id)) {
      # Pattern without mask: mul(matmul_result, scale)
      scores_id <- softmax_inp_id
    } else {
      next
    }

    # scores_id should be mul(matmul_result, scale)
    scores_node <- nodes[[as.character(scores_id)]]
    if (is.null(scores_node) || scores_node$op != "mul" ||
        length(scores_node$inputs) != 2L) next
    if (!single_consumer(scores_id)) next

    # One input to mul should be a matmul(Q, K^T), the other a scale constant
    mm_id <- NULL
    for (inp in scores_node$inputs) {
      inp_node <- nodes[[as.character(inp)]]
      if (!is.null(inp_node) && inp_node$op == "matmul") {
        mm_id <- inp
        break
      }
    }
    if (is.null(mm_id)) next

    mm_node <- nodes[[as.character(mm_id)]]
    if (length(mm_node$inputs) != 2L) next
    if (!single_consumer(mm_id)) next

    # mm_node is matmul(Q, K_transposed)
    q_id <- mm_node$inputs[1]
    kt_id <- mm_node$inputs[2]

    # kt should be a transpose of K (transpose with dims 3,4 or -1,-2)
    kt_node <- nodes[[as.character(kt_id)]]
    if (is.null(kt_node) || kt_node$op != "transpose") next

    d0 <- kt_node$attrs$dim0 %||% kt_node$attrs$arg1
    d1 <- kt_node$attrs$dim1 %||% kt_node$attrs$arg2
    # Accept transpose(3,4) for 1-based or transpose(-2,-1) for negative dims
    valid_transpose <- (!is.null(d0) && !is.null(d1)) &&
      ((d0 == 3L && d1 == 4L) || (d0 == 4L && d1 == 3L) ||
       (d0 == -2L && d1 == -1L) || (d0 == -1L && d1 == -2L))
    if (!valid_transpose) next

    # K is the input to the transpose
    k_id <- kt_node$inputs[1]

    # Build the SDPA node, reusing the final matmul's ID
    if (!is.null(mask_id)) {
      nodes[[id_str]] <- ir_node(id, "torch_sdpa",
                                  c(q_id, k_id, v_id, mask_id))
    } else {
      nodes[[id_str]] <- ir_node(id, "torch_sdpa",
                                  c(q_id, k_id, v_id))
    }
  }

  ir_graph(nodes, graph$input_ids, graph$output_ids)
}


#' Annotate Reduction Kernels
#'
#' Identifies IR nodes that can be compiled to dedicated Triton reduction
#' kernels: \code{softmax}, \code{log_softmax}, and \code{torch_layer_norm}.
#' Annotates them with a \code{reduction_kernel} attribute indicating the
#' kernel type. These are compiled separately from elementwise fusion groups.
#'
#' @param graph An ir_graph
#' @return A new ir_graph with reduction_kernel annotations
#' @noRd
annotate_reduction_kernels <- function(graph) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  nodes <- .clone_nodes(graph$nodes)

  for (id_str in names(nodes)) {
    node <- nodes[[id_str]]
    op <- node$op

    if (op %in% c("softmax", "nnf_softmax", "torch_softmax")) {
      # Row-wise softmax: compile to fused Triton reduction kernel.
      # Default dim=-1 (last dimension) which is the standard case.
      dim <- node$attrs$dim %||% node$attrs$arg1 %||% -1L
      if (dim == -1L) {
        nodes[[id_str]]$attrs$reduction_kernel <- "softmax"
      }

    } else if (op %in% c("log_softmax", "nnf_log_softmax")) {
      dim <- node$attrs$dim %||% node$attrs$arg1 %||% -1L
      if (dim == -1L) {
        nodes[[id_str]]$attrs$reduction_kernel <- "log_softmax"
      }

    } else if (op %in% c("torch_layer_norm", "nnf_layer_norm")) {
      # Full layer norm: mean, var, normalize, scale, shift in one kernel
      nodes[[id_str]]$attrs$reduction_kernel <- "layer_norm"
    }
  }

  ir_graph(nodes, graph$input_ids, graph$output_ids)
}


#' Run Optimization Pipeline
#'
#' Applies a sequence of optimization passes to an IR graph.
#' Default pipeline: constant_fold, dead_code_eliminate,
#' common_subexpr_eliminate, algebraic_simplify, dead_code_eliminate,
#' fusion_annotate.
#'
#' @param graph An ir_graph
#' @param passes List of pass functions. If NULL, uses default pipeline.
#' @return An optimized ir_graph
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()$sigmoid()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' optimize_graph(g)
#' }
#' @export
optimize_graph <- function(graph, passes = NULL) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  if (is.null(passes)) {
    passes <- list(
      decompose_high_level_ops,
      constant_fold,
      dead_code_eliminate,
      common_subexpr_eliminate,
      algebraic_simplify,
      dead_code_eliminate,
      detect_sdpa_patterns,
      dead_code_eliminate,
      fusion_annotate,
      annotate_reduction_kernels
    )
  }

  for (pass in passes) {
    graph <- pass(graph)
  }

  graph
}
