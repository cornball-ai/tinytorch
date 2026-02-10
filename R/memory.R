#' Memory Planning for IR Graphs
#'
#' Pre-plans memory allocations using shape inference and liveness analysis.
#' Applies greedy interval scheduling for buffer reuse and detects
#' in-place operation opportunities.

# Bytes per element for each dtype
.dtype_bytes <- function(dtype) {
  if (is.null(dtype)) return(4L)
  switch(dtype,
    float16 = 2L, bfloat16 = 2L,
    float32 = 4L, float = 4L,
    float64 = 8L, double = 8L,
    int8 = 1L, uint8 = 1L,
    int16 = 2L,
    int32 = 4L, int = 4L,
    int64 = 8L, long = 8L,
    bool = 1L,
    4L
  )
}

# Total bytes for a tensor
.tensor_bytes <- function(shape, dtype) {
  if (is.null(shape) || length(shape) == 0L) return(0)
  as.numeric(prod(shape)) * .dtype_bytes(dtype)
}

# Format bytes for display
.format_bytes <- function(bytes) {
  if (bytes >= 1024^3) return(sprintf("%.1f GB", bytes / 1024^3))
  if (bytes >= 1024^2) return(sprintf("%.1f MB", bytes / 1024^2))
  if (bytes >= 1024) return(sprintf("%.1f KB", bytes / 1024))
  sprintf("%d B", as.integer(bytes))
}

# Ops eligible for in-place execution
.inplace_ops <- c(
  "relu", "sigmoid", "tanh", "exp", "log", "log2", "log10",
  "sqrt", "abs", "neg", "sign", "floor", "ceil", "round", "trunc",
  "gelu", "silu", "leaky_relu", "elu",
  "add", "sub", "mul", "div", "pow", "remainder", "floor_div"
)


#' Detect In-Place Operation Opportunities
#'
#' For each eligible node, checks if it can write output directly into
#' an input's buffer (input has no remaining consumers and shapes match).
#'
#' @param graph An ir_graph with shape annotations
#' @param liveness Named list from analyze_liveness()
#' @return Named list of node_id_str -> input_node_id (integer)
#' @noRd
detect_in_place <- function(graph, liveness) {
  in_place <- list()

  ids <- sort(as.integer(names(graph$nodes)))

  for (id in ids) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]
    if (!node$op %in% .inplace_ops) next
    if (length(node$inputs) == 0L) next

    out_shape <- node$attrs$output_shape
    out_dtype <- node$attrs$output_dtype
    if (is.null(out_shape) || is.null(out_dtype)) next

    for (inp_id in node$inputs) {
      inp_str <- as.character(inp_id)
      inp_node <- graph$nodes[[inp_str]]
      if (is.null(inp_node)) next
      if (inp_node$op == "input") next
      if (inp_node$op == "constant") next

      # Input must die at this node
      lv <- liveness[[inp_str]]
      if (is.null(lv) || is.na(lv) || !is.finite(lv)) next
      if (lv != node$id) next

      # Shapes and dtypes must match
      inp_shape <- inp_node$attrs$output_shape
      inp_dtype <- inp_node$attrs$output_dtype
      if (is.null(inp_shape) || is.null(inp_dtype)) next
      if (!identical(as.integer(inp_shape), as.integer(out_shape))) next
      if (!identical(inp_dtype, out_dtype)) next

      in_place[[id_str]] <- inp_id
      break
    }
  }

  in_place
}


#' Plan Memory Allocation for an IR Graph
#'
#' Uses shape inference and liveness analysis to pre-plan all memory
#' allocations. Applies greedy interval scheduling (best-fit) for
#' buffer reuse and detects in-place operation opportunities.
#'
#' @param graph An ir_graph
#' @param input_shapes Named list of input shapes (used if graph lacks annotations)
#' @param input_dtypes Optional named list of input dtypes
#' @return A memory_plan object
#' @export
plan_memory <- function(graph, input_shapes = NULL, input_dtypes = NULL) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  # Ensure shape annotations
  has_shapes <- any(vapply(graph$nodes, function(n) {
    !is.null(n$attrs$output_shape)
  }, logical(1)))

  if (!has_shapes) {
    if (is.null(input_shapes)) {
      stop("Graph has no shape annotations and no input_shapes provided",
           call. = FALSE)
    }
    graph <- infer_shapes(graph, input_shapes, input_dtypes)
  }

  liveness <- analyze_liveness(graph)
  in_place <- detect_in_place(graph, liveness)

  ids <- sort(as.integer(names(graph$nodes)))

  # Compute byte sizes per node
  node_size <- list()
  for (id in ids) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]
    b <- .tensor_bytes(node$attrs$output_shape, node$attrs$output_dtype)
    if (b > 0) node_size[[id_str]] <- b
  }

  # Greedy buffer assignment
  assignments <- list()
  buf_sizes <- numeric()
  active <- list()        # buf_id_str -> last_use (numeric)
  free_pool <- integer()
  next_buf <- 1L
  naive_total <- 0

  for (id in ids) {
    id_str <- as.character(id)
    node <- graph$nodes[[id_str]]

    if (is.null(node_size[[id_str]])) next
    if (node$op %in% c("input", "constant")) next

    size <- node_size[[id_str]]
    last_use <- liveness[[id_str]]
    if (is.null(last_use) || is.na(last_use)) last_use <- as.numeric(id)

    naive_total <- naive_total + size

    # Free expired buffers
    expired <- character()
    for (buf_str in names(active)) {
      if (is.finite(active[[buf_str]]) && active[[buf_str]] < id) {
        expired <- c(expired, buf_str)
      }
    }
    for (buf_str in expired) {
      free_pool <- c(free_pool, as.integer(buf_str))
      active[[buf_str]] <- NULL
    }

    # In-place: reuse input's buffer
    if (!is.null(in_place[[id_str]])) {
      inp_str <- as.character(in_place[[id_str]])
      if (!is.null(assignments[[inp_str]])) {
        buf_id <- assignments[[inp_str]]
        assignments[[id_str]] <- buf_id
        buf_str <- as.character(buf_id)
        active[[buf_str]] <- max(active[[buf_str]] %||% 0, last_use)
        free_pool <- free_pool[free_pool != buf_id]
        next
      }
    }

    # Best-fit reuse from free pool
    reused <- FALSE
    if (length(free_pool) > 0) {
      best_idx <- NULL
      best_size <- Inf
      for (i in seq_along(free_pool)) {
        bs <- buf_sizes[free_pool[i]]
        if (bs >= size && bs < best_size) {
          best_idx <- i
          best_size <- bs
        }
      }
      if (!is.null(best_idx)) {
        buf_id <- free_pool[best_idx]
        free_pool <- free_pool[-best_idx]
        assignments[[id_str]] <- buf_id
        active[[as.character(buf_id)]] <- last_use
        reused <- TRUE
      }
    }

    # Allocate new buffer
    if (!reused) {
      buf_id <- next_buf
      next_buf <- next_buf + 1L
      buf_sizes[buf_id] <- size
      assignments[[id_str]] <- buf_id
      active[[as.character(buf_id)]] <- last_use
    }
  }

  # Build buffer summary
  buffers <- list()
  for (i in seq_along(buf_sizes)) {
    buffers[[as.character(i)]] <- list(size_bytes = buf_sizes[i])
  }

  total_bytes <- sum(buf_sizes)

  structure(
    list(
      assignments = assignments,
      buffers = buffers,
      in_place = in_place,
      total_bytes = total_bytes,
      naive_bytes = naive_total,
      n_buffers = length(buffers),
      reuse_pct = if (naive_total > 0) {
        round((1 - total_bytes / naive_total) * 100, 1)
      } else {
        0
      }
    ),
    class = "memory_plan"
  )
}


#' @export
print.memory_plan <- function(x, ...) {
  cat(sprintf("Memory Plan: %d buffer(s), %s total\n",
              x$n_buffers, .format_bytes(x$total_bytes)))
  cat(sprintf("  Naive: %s (%d allocations)\n",
              .format_bytes(x$naive_bytes),
              length(x$assignments)))
  if (x$naive_bytes > 0) {
    cat(sprintf("  Savings: %.1f%%", x$reuse_pct))
    n_inplace <- length(x$in_place)
    n_reuse <- length(x$assignments) - x$n_buffers - n_inplace
    if (n_reuse < 0) n_reuse <- 0L
    parts <- character()
    if (n_reuse > 0) parts <- c(parts, sprintf("%d reused", n_reuse))
    if (n_inplace > 0) parts <- c(parts, sprintf("%d in-place", n_inplace))
    if (length(parts) > 0) cat(sprintf(" (%s)", paste(parts, collapse = ", ")))
    cat("\n")
  }
  invisible(x)
}
