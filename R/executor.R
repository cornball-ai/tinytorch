#' Whole-Graph Executor for Optimized IR
#'
#' Runs a full optimization pipeline on an IR graph and executes it,
#' dispatching fused kernel groups to compiled SIMD kernels (CPU) or
#' Triton kernels (GPU via ariel) and non-fused ops to torch.
#'
#' Two-level API:
#' - `prepare_graph()` + `execute_prepared()`: explicit two-step for
#'   performance-sensitive code (prepare once, execute many times)
#' - `execute_optimized()`: convenient one-shot API with transparent
#'   auto-caching of prepared graphs

# Helper to wrap raw external pointer as torch_tensor
.wrap_result_tensor <- function(ptr) {
  class(ptr) <- "torch_tensor"
  ptr
}


# Topological sort: guarantees every node is executed after its inputs.
# Kahn's algorithm — O(V+E), no recursion.
.topo_sort <- function(graph) {
  nodes <- graph$nodes
  all_ids <- as.integer(names(nodes))

  # Build in-degree counts and adjacency (node -> dependents)
  in_deg <- integer(length(all_ids))
  names(in_deg) <- as.character(all_ids)
  dependents <- vector("list", length(all_ids))
  names(dependents) <- as.character(all_ids)

  for (id_str in names(nodes)) {
    node <- nodes[[id_str]]
    for (inp_id in node$inputs) {
      inp_str <- as.character(inp_id)
      if (inp_str %in% names(in_deg)) {
        in_deg[[id_str]] <- in_deg[[id_str]] + 1L
        dependents[[inp_str]] <- c(dependents[[inp_str]], as.integer(id_str))
      }
    }
  }

  # Start with zero in-degree nodes, ordered by ID for determinism
  queue <- sort(all_ids[in_deg == 0L])
  result <- integer(0)

  while (length(queue) > 0L) {
    id <- queue[[1L]]
    queue <- queue[-1L]
    result <- c(result, id)
    id_str <- as.character(id)
    for (dep in dependents[[id_str]]) {
      dep_str <- as.character(dep)
      in_deg[[dep_str]] <- in_deg[[dep_str]] - 1L
      if (in_deg[[dep_str]] == 0L) {
        # Insert sorted for deterministic order among ready nodes
        queue <- sort(c(queue, dep))
      }
    }
  }
  result
}


#' Detect Backend from Input Tensors
#'
#' @param inputs Named list of torch_tensors
#' @return "gpu" if any input is on CUDA and ariel is available, else "cpu"
#' @noRd
.detect_backend <- function(inputs) {
  has_cuda <- FALSE
  for (t in inputs) {
    if (inherits(t, "torch_tensor") && t$is_cuda) {
      has_cuda <- TRUE
      break
    }
  }
  if (has_cuda && requireNamespace("ariel", quietly = TRUE)) {
    "gpu"
  } else {
    "cpu"
  }
}


#' Dispatch a Single Torch Operation
#'
#' Calls the appropriate torch function for a given IR op.
#'
#' @param op Operation name
#' @param inputs List of torch_tensor inputs
#' @param attrs Named list of node attributes
#' @return A torch_tensor result
#' @noRd
dispatch_torch_op <- function(op, inputs, attrs = list()) {
  switch(op,
    # Unary activations
    relu = inputs[[1]]$relu(),
    sigmoid = inputs[[1]]$sigmoid(),
    tanh = inputs[[1]]$tanh(),
    exp = inputs[[1]]$exp(),
    log = inputs[[1]]$log(),
    log2 = inputs[[1]]$log2(),
    log10 = inputs[[1]]$log10(),
    sqrt = inputs[[1]]$sqrt(),
    abs = inputs[[1]]$abs(),
    neg = -inputs[[1]],
    sign = inputs[[1]]$sign(),
    floor = inputs[[1]]$floor(),
    ceil = inputs[[1]]$ceil(),
    round = inputs[[1]]$round(),
    trunc = inputs[[1]]$trunc(),
    silu = nnf_silu(inputs[[1]]),
    gelu = nnf_gelu(inputs[[1]]),
    leaky_relu = nnf_leaky_relu(inputs[[1]]),
    elu = nnf_elu(inputs[[1]]),
    softmax = nnf_softmax(inputs[[1]], dim = attrs$dim %||% -1L),
    log_softmax = nnf_log_softmax(inputs[[1]], dim = attrs$dim %||% -1L),
    contiguous = inputs[[1]]$contiguous(),
    clone = inputs[[1]]$clone(),
    detach = inputs[[1]]$detach(),

    # Binary ops
    add = inputs[[1]] + inputs[[2]],
    sub = inputs[[1]] - inputs[[2]],
    mul = inputs[[1]] * inputs[[2]],
    div = inputs[[1]] / inputs[[2]],
    pow = inputs[[1]] ^ inputs[[2]],
    remainder = inputs[[1]] %% inputs[[2]],
    floor_div = inputs[[1]] %/% inputs[[2]],

    # Comparison
    eq = inputs[[1]] == inputs[[2]],
    ne = inputs[[1]] != inputs[[2]],
    lt = inputs[[1]] < inputs[[2]],
    le = inputs[[1]] <= inputs[[2]],
    gt = inputs[[1]] > inputs[[2]],
    ge = inputs[[1]] >= inputs[[2]],

    # Matmul
    matmul = inputs[[1]]$matmul(inputs[[2]]),
    mm = inputs[[1]]$mm(inputs[[2]]),
    bmm = inputs[[1]]$bmm(inputs[[2]]),

    # Reductions
    sum = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) {
        inputs[[1]]$sum(as.integer(d), keepdim = isTRUE(attrs$keepdim))
      } else {
        inputs[[1]]$sum()
      }
    },
    mean = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) {
        inputs[[1]]$mean(as.integer(d), keepdim = isTRUE(attrs$keepdim))
      } else {
        inputs[[1]]$mean()
      }
    },

    # Shape ops
    reshape = inputs[[1]]$reshape(as.integer(attrs$shape %||% attrs$arg1)),
    view = inputs[[1]]$view(as.integer(attrs$shape %||% attrs$arg1)),
    transpose = {
      d0 <- as.integer(attrs$dim0 %||% attrs$arg1 %||% 1L)
      d1 <- as.integer(attrs$dim1 %||% attrs$arg2 %||% 2L)
      inputs[[1]]$transpose(d0, d1)
    },
    squeeze = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) inputs[[1]]$squeeze(as.integer(d))
      else inputs[[1]]$squeeze()
    },
    unsqueeze = inputs[[1]]$unsqueeze(as.integer(attrs$dim %||% attrs$arg1)),
    flatten = {
      sd <- as.integer(attrs$start_dim %||% attrs$arg1 %||% 1L)
      ed <- as.integer(attrs$end_dim %||% attrs$arg2 %||% -1L)
      inputs[[1]]$flatten(sd, ed)
    },

    # High-level torch ops (from traced modules)
    torch_linear = {
      bias <- if (length(inputs) >= 3L) inputs[[3]] else NULL
      torch_linear(inputs[[1]], inputs[[2]], bias)
    },
    torch_layer_norm = {
      # normalized_shape may be in attrs or as a constant input
      nshape <- attrs$normalized_shape %||% attrs$arg1
      weight <- NULL
      bias <- NULL
      if (is.null(nshape) && length(inputs) >= 2L && is.list(inputs[[2]])) {
        # normalized_shape came as a constant IR node (e.g., list(384L))
        nshape <- inputs[[2]]
        weight <- if (length(inputs) >= 3L) inputs[[3]] else NULL
        bias <- if (length(inputs) >= 4L) inputs[[4]] else NULL
      } else {
        weight <- if (length(inputs) >= 2L) inputs[[2]] else NULL
        bias <- if (length(inputs) >= 3L) inputs[[3]] else NULL
      }
      eps <- attrs$eps %||% 1e-5
      nnf_layer_norm(inputs[[1]], nshape, weight, bias, eps)
    },
    torch_gelu = nnf_gelu(inputs[[1]]),
    torch_silu = nnf_silu(inputs[[1]]),
    torch_sigmoid = torch_sigmoid(inputs[[1]]),
    torch_conv1d = {
      bias <- if (length(inputs) >= 3L) inputs[[3]] else NULL
      stride <- attrs$stride %||% 1L
      padding <- attrs$padding %||% 0L
      dilation <- attrs$dilation %||% 1L
      groups <- attrs$groups %||% 1L
      torch_conv1d(inputs[[1]], inputs[[2]], bias,
                    stride, padding, dilation, groups)
    },

    # Fallback: try method call on first input
    {
      if (length(inputs) >= 1L) {
        fn <- tryCatch(inputs[[1]][[op]], error = function(e) NULL)
        if (is.function(fn)) {
          if (length(inputs) == 1L) fn()
          else if (length(inputs) == 2L) fn(inputs[[2]])
          else do.call(fn, inputs[-1])
        } else {
          stop(sprintf("Unknown op: %s", op), call. = FALSE)
        }
      } else {
        stop(sprintf("Unknown op: %s (no inputs)", op), call. = FALSE)
      }
    }
  )
}


# ============================================================
# Execution cache
# ============================================================

.exec_cache <- new.env(parent = emptyenv())

# Build a structural fingerprint of a graph (constant for a given IR).
# Called once at prepare/trace time, not per-execution.
.graph_fingerprint <- function(graph) {
  ids <- sort(as.integer(names(graph$nodes)))
  parts <- vapply(ids, function(id) {
    node <- graph$nodes[[as.character(id)]]
    inp_str <- paste0(node$inputs, collapse = ",")
    paste0(id, ":", node$op, ":", inp_str)
  }, character(1))
  paste0(parts, collapse = "|")
}

# Build shape/dtype fingerprint from tensor list via C++.
# Single .Call() replaces N*(shape+dtype) R->C++ crossings.
.shapes_fingerprint <- function(inputs) {
  # Filter to only torch_tensor inputs (skip list constants etc.)
  tensor_inputs <- list()
  for (nm in sort(names(inputs))) {
    if (inherits(inputs[[nm]], "torch_tensor")) {
      tensor_inputs[[nm]] <- inputs[[nm]]
    }
  }
  if (length(tensor_inputs) == 0L) return("")
  .Call(cpp_tensor_shapes_key, tensor_inputs)
}

# Full cache key: graph_fp || shape_fp || backend
# For hot-path use, pre-compute graph_fp and pass it in.
.make_exec_cache_key <- function(graph, inputs, backend = "cpu",
                                  graph_fp = NULL) {
  if (is.null(graph_fp)) graph_fp <- .graph_fingerprint(graph)
  shape_fp <- .shapes_fingerprint(inputs)
  paste0(graph_fp, "||", shape_fp, "||", backend)
}


# ============================================================
# Compiled executor: code generation at prepare time
# ============================================================

# Generate an R expression for a single torch op.
# Called once at prepare time per compute node.
.gen_op_expr <- function(op, inp, attrs) {
  switch(op,
    # Unary (method)
    relu = bquote(.(inp[[1]])$relu()),
    sigmoid = bquote(.(inp[[1]])$sigmoid()),
    tanh = bquote(.(inp[[1]])$tanh()),
    exp = bquote(.(inp[[1]])$exp()),
    log = bquote(.(inp[[1]])$log()),
    log2 = bquote(.(inp[[1]])$log2()),
    log10 = bquote(.(inp[[1]])$log10()),
    sqrt = bquote(.(inp[[1]])$sqrt()),
    abs = bquote(.(inp[[1]])$abs()),
    neg = bquote(-.(inp[[1]])),
    sign = bquote(.(inp[[1]])$sign()),
    floor = bquote(.(inp[[1]])$floor()),
    ceil = bquote(.(inp[[1]])$ceil()),
    round = bquote(.(inp[[1]])$round()),
    trunc = bquote(.(inp[[1]])$trunc()),
    contiguous = bquote(.(inp[[1]])$contiguous()),
    clone = bquote(.(inp[[1]])$clone()),
    detach = bquote(.(inp[[1]])$detach()),

    # Unary (function)
    silu = bquote(nnf_silu(.(inp[[1]]))),
    gelu = bquote(nnf_gelu(.(inp[[1]]))),
    leaky_relu = bquote(nnf_leaky_relu(.(inp[[1]]))),
    elu = bquote(nnf_elu(.(inp[[1]]))),
    softmax = bquote(nnf_softmax(.(inp[[1]]), dim = .(attrs$dim %||% -1L))),
    log_softmax = bquote(nnf_log_softmax(.(inp[[1]]), dim = .(attrs$dim %||% -1L))),

    # Binary
    add = bquote(.(inp[[1]]) + .(inp[[2]])),
    sub = bquote(.(inp[[1]]) - .(inp[[2]])),
    mul = bquote(.(inp[[1]]) * .(inp[[2]])),
    div = bquote(.(inp[[1]]) / .(inp[[2]])),
    pow = bquote(.(inp[[1]]) ^ .(inp[[2]])),
    remainder = bquote(.(inp[[1]]) %% .(inp[[2]])),
    floor_div = bquote(.(inp[[1]]) %/% .(inp[[2]])),

    # Comparison
    eq = bquote(.(inp[[1]]) == .(inp[[2]])),
    ne = bquote(.(inp[[1]]) != .(inp[[2]])),
    lt = bquote(.(inp[[1]]) < .(inp[[2]])),
    le = bquote(.(inp[[1]]) <= .(inp[[2]])),
    gt = bquote(.(inp[[1]]) > .(inp[[2]])),
    ge = bquote(.(inp[[1]]) >= .(inp[[2]])),

    # Matmul
    matmul = bquote(.(inp[[1]])$matmul(.(inp[[2]]))),
    mm = bquote(.(inp[[1]])$mm(.(inp[[2]]))),
    bmm = bquote(.(inp[[1]])$bmm(.(inp[[2]]))),

    # Reductions
    sum = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) {
        bquote(.(inp[[1]])$sum(.(as.integer(d)),
                                keepdim = .(isTRUE(attrs$keepdim))))
      } else {
        bquote(.(inp[[1]])$sum())
      }
    },
    mean = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) {
        bquote(.(inp[[1]])$mean(.(as.integer(d)),
                                 keepdim = .(isTRUE(attrs$keepdim))))
      } else {
        bquote(.(inp[[1]])$mean())
      }
    },

    # Shape ops
    transpose = {
      d0 <- as.integer(attrs$dim0 %||% attrs$arg1 %||% 1L)
      d1 <- as.integer(attrs$dim1 %||% attrs$arg2 %||% 2L)
      bquote(.(inp[[1]])$transpose(.(d0), .(d1)))
    },
    reshape = bquote(.(inp[[1]])$reshape(.(as.integer(attrs$shape %||% attrs$arg1)))),
    view = bquote(.(inp[[1]])$view(.(as.integer(attrs$shape %||% attrs$arg1)))),
    squeeze = {
      d <- attrs$dim %||% attrs$arg1
      if (!is.null(d)) bquote(.(inp[[1]])$squeeze(.(as.integer(d))))
      else bquote(.(inp[[1]])$squeeze())
    },
    unsqueeze = bquote(.(inp[[1]])$unsqueeze(.(as.integer(attrs$dim %||% attrs$arg1)))),
    flatten = {
      sd <- as.integer(attrs$start_dim %||% attrs$arg1 %||% 1L)
      ed <- as.integer(attrs$end_dim %||% attrs$arg2 %||% -1L)
      bquote(.(inp[[1]])$flatten(.(sd), .(ed)))
    },

    # High-level torch ops (from traced modules)
    torch_linear = {
      if (length(inp) >= 3L) {
        bquote(torch_linear(.(inp[[1]]), .(inp[[2]]), .(inp[[3]])))
      } else {
        bquote(torch_linear(.(inp[[1]]), .(inp[[2]])))
      }
    },
    torch_layer_norm = {
      nshape <- attrs$normalized_shape %||% attrs$arg1
      eps <- attrs$eps %||% 1e-5
      if (!is.null(nshape)) {
        if (length(inp) >= 3L) {
          bquote(nnf_layer_norm(.(inp[[1]]), .(nshape),
                                 .(inp[[2]]), .(inp[[3]]), .(eps)))
        } else if (length(inp) >= 2L) {
          bquote(nnf_layer_norm(.(inp[[1]]), .(nshape),
                                 .(inp[[2]]), eps = .(eps)))
        } else {
          bquote(nnf_layer_norm(.(inp[[1]]), .(nshape), eps = .(eps)))
        }
      } else {
        # normalized_shape as constant input (inp[[2]])
        if (length(inp) >= 4L) {
          bquote(nnf_layer_norm(.(inp[[1]]), .(inp[[2]]),
                                 .(inp[[3]]), .(inp[[4]]), .(eps)))
        } else if (length(inp) >= 3L) {
          bquote(nnf_layer_norm(.(inp[[1]]), .(inp[[2]]),
                                 .(inp[[3]]), eps = .(eps)))
        } else {
          bquote(nnf_layer_norm(.(inp[[1]]), .(inp[[2]]), eps = .(eps)))
        }
      }
    },
    torch_gelu = bquote(nnf_gelu(.(inp[[1]]))),
    torch_silu = bquote(nnf_silu(.(inp[[1]]))),
    torch_sigmoid = bquote(torch_sigmoid(.(inp[[1]]))),
    torch_conv1d = {
      bias_expr <- if (length(inp) >= 3L) inp[[3]] else quote(NULL)
      stride <- attrs$stride %||% 1L
      padding <- attrs$padding %||% 0L
      dilation <- attrs$dilation %||% 1L
      groups <- attrs$groups %||% 1L
      bquote(torch_conv1d(.(inp[[1]]), .(inp[[2]]), .(bias_expr),
                           .(stride), .(padding), .(dilation), .(groups)))
    },

    # Fallback: dispatch through the interpreter
    {
      inp_list <- as.call(c(list(as.name("list")), inp))
      bquote(dispatch_torch_op(.(op), .(inp_list), .(attrs)))
    }
  )
}


# Compile a prepared_graph into a single R function.
# Walks the graph once at prepare time and generates direct R code
# with local variables, inlined torch calls, and pre-bound kernels.
# The generated function has no loop, no switch dispatch, no tryCatch
# per node, and no hash-keyed value lookups.
.compile_fast_executor <- function(prepared, target_device = NULL) {
  graph <- prepared$graph
  kernels <- prepared$kernels
  matmul_epilogues <- prepared$matmul_epilogues %||% list()
  fused_node_set <- prepared$fused_node_set
  exec_order <- prepared$exec_order

  vn <- function(id) as.name(paste0("v", id))

  # Closure environment: constants and kernel fns bound here
  fn_env <- new.env(parent = asNamespace("Rtorch"))
  fn_env$dispatch_torch_op <- dispatch_torch_op
  fn_env$.wrap_result_tensor <- .wrap_result_tensor

  body_exprs <- list()

  # Map fusion group entry IDs to kernel info
  fusion_entries <- new.env(parent = emptyenv())
  for (gid_str in names(kernels)) {
    k <- kernels[[gid_str]]
    entry_id <- as.character(min(k$group_node_ids))
    fusion_entries[[entry_id]] <- k
  }

  # Map matmul epilogue entry IDs (keyed by matmul node ID)
  matmul_epi_entries <- new.env(parent = emptyenv())
  for (mm_str in names(matmul_epilogues)) {
    matmul_epi_entries[[mm_str]] <- matmul_epilogues[[mm_str]]
  }

  for (id in exec_order) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]

    # Input: v1 <- inputs[["x"]]
    if (node$op == "input") {
      nm <- node$attrs$name
      if (!is.null(nm)) {
        body_exprs[[length(body_exprs) + 1L]] <- bquote(
          .(vn(id)) <- inputs[[.(nm)]]
        )
      }
      next
    }

    # Constant: pre-compute and bind in closure env
    if (node$op == "constant") {
      cname <- paste0(".c", id)
      val <- node$attrs$value
      if (is.numeric(val) || is.logical(val)) {
        ct <- torch_tensor(val)
        if (!is.null(target_device)) ct <- ct$to(device = target_device)
        fn_env[[cname]] <- ct
      } else {
        fn_env[[cname]] <- val
      }
      body_exprs[[length(body_exprs) + 1L]] <- bquote(
        .(vn(id)) <- .(as.name(cname))
      )
      next
    }

    # Matmul epilogue: fused matmul+bias+epilogue kernel
    if (exists(id_str, envir = matmul_epi_entries, inherits = FALSE)) {
      k <- matmul_epi_entries[[id_str]]
      kname <- paste0(".mk", id)
      fn_env[[kname]] <- k$call_fn
      out_var <- vn(k$output_id)
      # call_fn(A, B, bias) — 3 inputs from external_input_ids
      call_args <- lapply(k$external_input_ids, function(eid) vn(eid))
      kcall <- as.call(c(list(as.name(kname)), call_args))
      body_exprs[[length(body_exprs) + 1L]] <- bquote(
        .(out_var) <- .(kcall)
      )
      next
    }

    # Fusion group entry: kernel call
    if (exists(id_str, envir = fusion_entries, inherits = FALSE)) {
      k <- fusion_entries[[id_str]]
      kname <- paste0(".k", id)
      fn_env[[kname]] <- k$call_fn
      is_gpu <- isTRUE(k$gpu)
      out_var <- vn(k$output_id)

      call_args <- lapply(k$external_input_ids, function(eid) vn(eid))
      kcall <- as.call(c(list(as.name(kname)), call_args))

      if (is_gpu) {
        body_exprs[[length(body_exprs) + 1L]] <- bquote(
          .(out_var) <- .(kcall)
        )
      } else {
        body_exprs[[length(body_exprs) + 1L]] <- bquote(
          .(out_var) <- .wrap_result_tensor(.(kcall))
        )
      }
      next
    }

    # Skip fused interior nodes
    if (id_str %in% fused_node_set) next

    # Regular compute: generate inlined torch call
    inp_vars <- lapply(node$inputs, function(iid) vn(iid))
    expr <- .gen_op_expr(node$op, inp_vars, node$attrs)
    body_exprs[[length(body_exprs) + 1L]] <- bquote(
      .(vn(id)) <- .(expr)
    )
  }

  # Return output
  if (length(graph$output_ids) == 1L) {
    body_exprs[[length(body_exprs) + 1L]] <- vn(graph$output_ids[[1]])
  } else {
    out_exprs <- lapply(graph$output_ids, function(oid) vn(oid))
    body_exprs[[length(body_exprs) + 1L]] <- as.call(
      c(list(as.name("list")), out_exprs)
    )
  }

  # Build function(inputs) { ... }
  fn_body <- as.call(c(list(as.name("{")), body_exprs))
  fn <- eval(call("function", as.pairlist(alist(inputs = )), fn_body))
  environment(fn) <- fn_env
  fn
}


#' Prepare an IR Graph for Fast Execution
#'
#' Performs all expensive optimization, shape inference, and fusion
#' compilation once. Returns a prepared execution plan that can be
#' executed many times via \code{execute_prepared()}.
#'
#' @param graph An ir_graph
#' @param example_inputs Named list of input torch_tensors (used for
#'   shape inference and fusion compilation)
#' @param optimize Logical, run optimization passes (default TRUE)
#' @param fuse Logical, compile fusion groups to kernels (default TRUE)
#' @param fuse_matmul_epilogues Logical, fuse matmul+bias+activation on GPU (default FALSE)
#' @param backend Character: "auto", "gpu", or "cpu". "auto" detects
#'   from input tensors and ariel availability.
#' @return A \code{prepared_graph} object
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' pg <- prepare_graph(g, list(x = torch_randn(c(2, 3))))
#' }
#' @export
prepare_graph <- function(graph, example_inputs, optimize = TRUE, fuse = TRUE,
                           fuse_matmul_epilogues = FALSE, backend = "auto") {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)
  if (!is.list(example_inputs)) {
    stop("example_inputs must be a named list of tensors", call. = FALSE)
  }

  # Resolve backend
  if (backend == "auto") {
    backend <- .detect_backend(example_inputs)
  }

  # 1. Optimize -- once
  if (optimize) {
    graph <- optimize_graph(graph)
  }

  # 2. Infer shapes -- once
  input_shapes <- list()
  input_dtypes <- list()
  for (inp_id in graph$input_ids) {
    nm <- graph$nodes[[as.character(inp_id)]]$attrs$name
    if (!is.null(nm) && nm %in% names(example_inputs)) {
      t <- example_inputs[[nm]]
      if (inherits(t, "torch_tensor")) {
        input_shapes[[nm]] <- as.integer(t$shape)
        dt <- as.character(t$dtype)
        input_dtypes[[nm]] <- switch(dt,
          Float = "float32", Double = "float64",
          Int = "int32", Long = "int64", "float32")
      }
    }
  }
  if (length(input_shapes) > 0L) {
    graph <- tryCatch(
      infer_shapes(graph, input_shapes, input_dtypes),
      error = function(e) graph
    )
  }

  # 3. Compile fusion groups -- once
  kernels <- list()
  fused_node_set <- character()
  if (fuse) {
    group_ids <- get_fusion_groups(graph)
    for (gid in group_ids) {
      k <- NULL
      if (backend == "gpu") {
        # Try GPU compilation first
        k <- tryCatch(
          compile_fusion_group_gpu(graph, gid),
          error = function(e) NULL
        )
      }
      if (is.null(k)) {
        # Fall back to CPU SIMD
        k <- tryCatch(compile_fusion_group(graph, gid), error = function(e) NULL)
        # CPU kernels require same-sized inputs (no broadcast).
        # Skip if external inputs have mismatched shapes.
        if (!is.null(k) && !isTRUE(k$gpu)) {
          shapes <- list()
          skip <- FALSE
          for (eid in k$external_input_ids) {
            nm <- graph$nodes[[as.character(eid)]]$attrs$name
            if (!is.null(nm) && nm %in% names(example_inputs)) {
              t <- example_inputs[[nm]]
              if (inherits(t, "torch_tensor")) {
                shapes <- c(shapes, list(as.integer(t$shape)))
              }
            }
          }
          if (length(shapes) >= 2L) {
            ref_numel <- prod(shapes[[1]])
            for (s in shapes[-1]) {
              if (prod(s) != ref_numel) { skip <- TRUE; break }
            }
          }
          if (skip) k <- NULL
        }
      }
      if (!is.null(k)) {
        kernels[[as.character(gid)]] <- k
        fused_node_set <- c(fused_node_set, as.character(k$group_node_ids))
      }
    }
  }

  # 3b. Detect and compile matmul epilogue patterns (GPU only)
  # Disabled by default: cuBLAS matmul + separate elementwise kernel is currently
  # faster than our Triton matmul with fused epilogue. Our tiled matmul doesn't
  # use tensor cores or auto-tuned tile sizes. Enable when the Triton matmul
  # kernel is competitive with cuBLAS.
  matmul_epilogues <- list()
  if (fuse && fuse_matmul_epilogues && backend == "gpu") {
    patterns <- tryCatch(detect_matmul_epilogues(graph), error = function(e) list())
    for (pat in patterns) {
      k <- tryCatch(
        compile_matmul_epilogue_gpu(pat),
        error = function(e) NULL
      )
      if (!is.null(k)) {
        # Store keyed by matmul node ID (the entry point for this kernel)
        matmul_epilogues[[as.character(pat$matmul_id)]] <- k
        # Add matmul + all fusion group nodes to fused set
        fused_node_set <- c(fused_node_set,
                             as.character(pat$matmul_id),
                             as.character(pat$all_node_ids))
        # Remove the fusion group from regular kernels (it's handled)
        kernels[[as.character(pat$group_id)]] <- NULL
      }
    }
  }

  # 4. Pre-compute execution order (topological sort)
  exec_order <- .topo_sort(graph)

  # 5. Build input name -> id_str mapping
  input_map <- list()  # name -> id_str
  for (inp_id in graph$input_ids) {
    id_str <- as.character(inp_id)
    nm <- graph$nodes[[id_str]]$attrs$name
    if (!is.null(nm)) {
      input_map[[nm]] <- id_str
    }
  }

  prepared <- structure(list(
    graph = graph,
    kernels = kernels,
    matmul_epilogues = matmul_epilogues,
    fused_node_set = fused_node_set,
    exec_order = exec_order,
    input_map = input_map,
    backend = backend
  ), class = "prepared_graph")

  # 6. Compile fast executor: generates a specialized R function
  #    with inlined torch calls, no per-node loop/switch/tryCatch.
  target_device <- NULL
  if (backend == "gpu") {
    for (t in example_inputs) {
      if (inherits(t, "torch_tensor") && t$is_cuda) {
        target_device <- t$device
        break
      }
    }
  }
  prepared$fast_fn <- tryCatch(
    .compile_fast_executor(prepared, target_device),
    error = function(e) NULL
  )

  prepared
}


#' Execute a Prepared Graph
#'
#' Fast execution path using a pre-computed execution plan from
#' \code{prepare_graph()}. No optimization or compilation overhead --
#' just runs the execution loop.
#'
#' @param prepared A \code{prepared_graph} object from \code{prepare_graph()}
#' @param inputs Named list of input torch_tensors
#' @param verbose Logical, print execution info
#' @return A torch_tensor (or list of tensors for multi-output graphs)
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' pg <- prepare_graph(g, list(x = torch_randn(c(2, 3))))
#' execute_prepared(pg, list(x = torch_randn(c(2, 3))))
#' }
#' @export
execute_prepared <- function(prepared, inputs, verbose = FALSE) {
  if (!inherits(prepared, "prepared_graph")) {
    stop("Expected a prepared_graph from prepare_graph()", call. = FALSE)
  }

  # Fast path: compiled executor (no loop, no switch, no tryCatch per node)
  if (!is.null(prepared$fast_fn) && !verbose) {
    return(prepared$fast_fn(inputs))
  }

  graph <- prepared$graph
  kernels <- prepared$kernels
  matmul_epilogues <- prepared$matmul_epilogues %||% list()
  fused_node_set <- prepared$fused_node_set
  input_map <- prepared$input_map
  backend <- prepared$backend %||% "cpu"

  # Initialize input values via pre-computed input_map
  values <- list()
  for (nm in names(input_map)) {
    if (nm %in% names(inputs)) {
      values[[input_map[[nm]]]] <- inputs[[nm]]
    }
  }

  # Determine target device for constants
  target_device <- NULL
  if (backend == "gpu") {
    for (t in inputs) {
      if (inherits(t, "torch_tensor") && t$is_cuda) {
        target_device <- t$device
        break
      }
    }
  }

  for (id in prepared$exec_order) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]

    # Skip already computed (inputs)
    if (!is.null(values[[id_str]])) next

    # Matmul epilogue: fused matmul+bias+epilogue
    if (id_str %in% names(matmul_epilogues)) {
      k <- matmul_epilogues[[id_str]]
      ext_inputs <- lapply(k$external_input_ids, function(eid) {
        values[[as.character(eid)]]
      })
      result <- tryCatch(
        do.call(k$call_fn, ext_inputs),
        error = function(e) {
          if (verbose) message(sprintf("  Matmul epilogue failed: %s",
                                       conditionMessage(e)))
          NULL
        }
      )
      if (!is.null(result)) {
        values[[as.character(k$output_id)]] <- result
        if (verbose) {
          message(sprintf("  Matmul+bias+%s fused (GPU)",
                          paste(k$epilogue_ops %||% "none", collapse = "+")))
        }
        next
      }
    }

    # Skip nodes handled by a fusion group (not the entry point)
    if (id_str %in% fused_node_set) {
      gid <- node$attrs$fusion_group
      if (!is.null(gid) && as.character(gid) %in% names(kernels)) {
        k <- kernels[[as.character(gid)]]
        if (id == min(k$group_node_ids)) {
          ext_inputs <- lapply(k$external_input_ids, function(eid) {
            values[[as.character(eid)]]
          })
          is_gpu_kernel <- isTRUE(k$gpu)
          result <- tryCatch({
            raw <- do.call(k$call_fn, ext_inputs)
            # GPU kernels return torch_tensor directly; CPU returns raw pointer
            if (is_gpu_kernel) raw else .wrap_result_tensor(raw)
          }, error = function(e) {
            if (verbose) message(sprintf("  Fusion failed: %s", conditionMessage(e)))
            NULL
          })

          if (!is.null(result)) {
            values[[as.character(k$output_id)]] <- result
            if (verbose) {
              tag <- if (is_gpu_kernel) "GPU" else "CPU"
              message(sprintf("  Fused group %d: %d ops -> 1 %s kernel",
                              gid, length(k$group_node_ids), tag))
            }
            next
          }
          # Fall through to non-fused execution on failure
        } else {
          next  # Interior node of a group, skip
        }
      }
    }

    # Handle constants
    if (node$op == "constant") {
      val <- node$attrs$value
      if (is.numeric(val) || is.logical(val)) {
        ct <- torch_tensor(val)
        if (!is.null(target_device)) {
          ct <- ct$to(device = target_device)
        }
        values[[id_str]] <- ct
      } else {
        # Non-tensor constants (e.g., list(384L) for normalized_shape)
        values[[id_str]] <- val
      }
      next
    }

    # Skip input nodes without values
    if (node$op == "input") next

    # Gather input tensors
    inp_tensors <- lapply(node$inputs, function(inp_id) {
      values[[as.character(inp_id)]]
    })

    # Check all inputs are available
    if (any(vapply(inp_tensors, is.null, logical(1)))) {
      if (verbose) message(sprintf("  Skip %%%d (%s): missing inputs", id, node$op))
      next
    }

    # Dispatch to torch
    result <- tryCatch(
      dispatch_torch_op(node$op, inp_tensors, node$attrs),
      error = function(e) {
        if (verbose) message(sprintf("  Error %%%d (%s): %s",
                                    id, node$op, conditionMessage(e)))
        NULL
      }
    )
    values[[id_str]] <- result

    if (verbose) {
      message(sprintf("  %%%d = %s", id, node$op))
    }
  }

  # Return output(s)
  outputs <- lapply(graph$output_ids, function(oid) {
    values[[as.character(oid)]]
  })

  if (length(outputs) == 1L) outputs[[1]] else outputs
}


#' Execute an Optimized IR Graph
#'
#' Runs the full optimization pipeline (constant folding, DCE, CSE,
#' algebraic simplification, fusion annotation) then executes the graph.
#' Fusion groups are compiled to SIMD-vectorized C++ kernels (CPU) or
#' Triton GPU kernels (via ariel); non-fused ops are dispatched to torch.
#'
#' Automatically caches prepared graphs keyed by graph structure,
#' input shapes, and backend. For single-op graphs, uses eager fallback
#' to avoid all pipeline overhead.
#'
#' For maximum performance in tight loops, use \code{prepare_graph()}
#' and \code{execute_prepared()} directly.
#'
#' @param graph An ir_graph
#' @param inputs Named list of input torch_tensors
#' @param optimize Logical, run optimization passes (default TRUE)
#' @param fuse Logical, compile fusion groups to kernels (default TRUE)
#' @param fuse_matmul_epilogues Logical, fuse matmul+bias+activation on GPU (default FALSE)
#' @param backend Character: "auto", "gpu", or "cpu". "auto" detects
#'   from input tensors and ariel availability.
#' @param verbose Logical, print execution info
#' @return A torch_tensor (or list of tensors for multi-output graphs)
#' @examples
#' \donttest{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' execute_optimized(g, list(x = torch_randn(c(2, 3))))
#' }
#' @export
execute_optimized <- function(graph, inputs, optimize = TRUE, fuse = TRUE,
                              fuse_matmul_epilogues = FALSE,
                              backend = "auto", verbose = FALSE) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)
  if (!is.list(inputs)) stop("inputs must be a named list of tensors", call. = FALSE)

  # Resolve backend early for cache key
  resolved_backend <- if (backend == "auto") .detect_backend(inputs) else backend

  nodes <- graph$nodes

  # Eager fallback: for small graphs, check if it's a single-op graph
  # and dispatch directly without any pipeline overhead.
  n_nodes <- length(nodes)
  if (n_nodes <= 4L) {
    # Small graph: find the compute node directly
    compute_node <- NULL
    for (id_str in names(nodes)) {
      nd <- nodes[[id_str]]
      if (nd$op != "input" && nd$op != "constant") {
        if (!is.null(compute_node)) {
          compute_node <- NULL  # More than one compute node
          break
        }
        compute_node <- nd
      }
    }
    if (!is.null(compute_node)) {
      # Single compute op: resolve inputs directly and dispatch
      inp_tensors <- vector("list", length(compute_node$inputs))
      for (j in seq_along(compute_node$inputs)) {
        iid <- as.character(compute_node$inputs[[j]])
        inp_node <- nodes[[iid]]
        if (inp_node$op == "input") {
          inp_tensors[[j]] <- inputs[[inp_node$attrs$name]]
        } else if (inp_node$op == "constant") {
          inp_tensors[[j]] <- torch_tensor(inp_node$attrs$value)
        }
      }
      return(dispatch_torch_op(compute_node$op, inp_tensors, compute_node$attrs))
    }
  }

  # Auto-cache: check for a cached prepared graph
  cache_key <- .make_exec_cache_key(graph, inputs, resolved_backend)
  prepared <- .exec_cache[[cache_key]]

  if (is.null(prepared)) {
    # Cache miss: run full pipeline
    prepared <- prepare_graph(graph, inputs, optimize = optimize, fuse = fuse,
                               fuse_matmul_epilogues = fuse_matmul_epilogues,
                               backend = resolved_backend)
    .exec_cache[[cache_key]] <- prepared
    if (verbose) message("Cache miss: prepared and cached graph")
  } else {
    if (verbose) message("Cache hit: using prepared graph")
  }

  execute_prepared(prepared, inputs, verbose = verbose)
}


#' Clear the Execution Cache
#'
#' Removes all cached prepared graphs from the auto-cache used by
#' \code{execute_optimized()}.
#'
#' @return Integer, number of entries cleared (invisibly)
#' @examples
#' clear_exec_cache()
#' @export
clear_exec_cache <- function() {
  nms <- ls(.exec_cache)
  n <- length(nms)
  rm(list = nms, envir = .exec_cache)
  invisible(n)
}


#' Execution Cache Statistics
#'
#' Returns information about the auto-cache used by
#' \code{execute_optimized()}.
#'
#' @return A list with \code{n_cached} (number of cached prepared graphs)
#' @examples
#' exec_cache_stats()
#' @export
exec_cache_stats <- function() {
  list(n_cached = length(ls(.exec_cache)))
}
