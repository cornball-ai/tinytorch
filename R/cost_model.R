#' Fusion Cost Model
#'
#' Estimates whether fusing a group of IR nodes into a single kernel is
#' profitable. Adapted from TorchInductor's three-factor scoring.
#'
#' @name cost_model
NULL


#' Estimate Fusion Cost
#'
#' Evaluates whether a fusion group is profitable by estimating memory traffic
#' savings. A fusion of N ops on a tensor of size S saves approximately
#' \code{(N-1) * S * dtype_bytes * 2} bytes of memory traffic (each eliminated
#' intermediate avoids one read + one write). Fusion is profitable when savings
#' exceed kernel launch overhead.
#'
#' @param graph An ir_graph
#' @param node_ids Integer vector of node IDs in the fusion group
#' @param min_elements Minimum tensor elements for fusion to be profitable.
#'   Below this threshold, kernel launch overhead exceeds memory savings.
#'   Default 1024L.
#' @param launch_overhead_us Estimated kernel launch overhead in microseconds.
#'   Default 5.0 for GPU, ~1.0 for CPU SIMD.
#' @return A list with components:
#'   \describe{
#'     \item{profitable}{Logical: is fusion worth it?}
#'     \item{n_ops}{Number of ops in the group}
#'     \item{n_elements}{Estimated tensor elements (NULL if shapes unknown)}
#'     \item{memory_saved_bytes}{Estimated bytes of memory traffic saved}
#'     \item{score}{Numeric fusion score (higher = more profitable)}
#'   }
#' @noRd
estimate_fusion_cost <- function(graph, node_ids,
                                  min_elements = 1024L,
                                  launch_overhead_us = 5.0) {
  nodes <- graph$nodes
  n_ops <- length(node_ids)

  # Single op: never profitable to fuse alone
  if (n_ops < 2L) {
    return(list(profitable = FALSE, n_ops = n_ops, n_elements = NULL,
                memory_saved_bytes = 0, score = 0))
  }

  # Try to get shape from any node in the group
  n_elements <- NULL
  dtype_bytes <- 4L  # default float32
  for (nid in node_ids) {
    node <- nodes[[as.character(nid)]]
    shape <- node$attrs$output_shape
    if (!is.null(shape) && length(shape) > 0L) {
      n_elements <- prod(as.integer(shape))
      # Check dtype
      dt <- node$attrs$output_dtype
      if (!is.null(dt)) {
        dtype_bytes <- switch(dt,
          float16 = 2L, float32 = 4L, float64 = 8L,
          int8 = 1L, int16 = 2L, int32 = 4L, int64 = 8L,
          4L  # default
        )
      }
      break
    }
  }

  # If shapes unknown, assume profitable (can't prove otherwise)
  if (is.null(n_elements)) {
    return(list(profitable = TRUE, n_ops = n_ops, n_elements = NULL,
                memory_saved_bytes = NA_real_, score = as.numeric(n_ops)))
  }

  # Too small: kernel launch overhead dominates
  if (n_elements < min_elements) {
    return(list(profitable = FALSE, n_ops = n_ops, n_elements = n_elements,
                memory_saved_bytes = 0, score = 0))
  }

  # Memory savings: each eliminated intermediate avoids one read + one write
  # N ops fused saves (N-1) intermediates
  memory_saved <- as.numeric(n_ops - 1L) * n_elements * dtype_bytes * 2

  # Score: memory_saved normalized by launch overhead
  # At 10 GB/s memory bandwidth, 5us launch overhead = 50KB breakeven
  score <- memory_saved / (launch_overhead_us * 1e-6 * 10e9)

  list(
    profitable = TRUE,
    n_ops = n_ops,
    n_elements = n_elements,
    memory_saved_bytes = memory_saved,
    score = score
  )
}


#' Estimate Reduction Kernel Cost
#'
#' Evaluates whether compiling a standalone reduction (softmax, layer_norm)
#' to a Triton kernel is profitable. Standalone reductions are typically
#' SLOWER than torch's built-in implementations due to R call overhead.
#' They are only profitable when fused with adjacent elementwise ops.
#'
#' @param graph An ir_graph
#' @param node_id Integer ID of the reduction node
#' @param consumers Named list mapping node ID strings to vectors of consumer
#'   node IDs
#' @return A list with \code{profitable} (logical) and \code{reason} (character)
#' @noRd
estimate_reduction_cost <- function(graph, node_id, consumers) {
  node_str <- as.character(node_id)
  node <- graph$nodes[[node_str]]

  # Check if the reduction has elementwise consumers that could be fused in
  consumer_ids <- consumers[[node_str]]
  if (is.null(consumer_ids) || length(consumer_ids) == 0L) {
    return(list(profitable = FALSE,
                reason = "standalone reduction, no consumers to fuse"))
  }

  elementwise_ops <- c(
    "relu", "sigmoid", "tanh", "exp", "log", "log2", "log10",
    "sqrt", "rsqrt", "abs", "neg", "sign", "floor", "ceil", "round", "trunc",
    "sin", "cos",
    "gelu", "silu", "leaky_relu", "elu",
    "add", "sub", "mul", "div", "pow", "remainder", "floor_div"
  )

  # If ANY consumer is an elementwise op, there's fusion potential
  has_elementwise_consumer <- FALSE
  for (cid in consumer_ids) {
    cnode <- graph$nodes[[as.character(cid)]]
    if (!is.null(cnode) && cnode$op %in% elementwise_ops) {
      has_elementwise_consumer <- TRUE
      break
    }
  }

  if (has_elementwise_consumer) {
    return(list(profitable = TRUE,
                reason = "reduction has elementwise consumers for fusion"))
  }

  # Standalone reduction: not profitable for Triton
  # Benchmarks: softmax 0.04x, layer_norm 0.50x vs torch built-in
  list(profitable = FALSE,
       reason = "standalone reduction, no elementwise fusion opportunity")
}
