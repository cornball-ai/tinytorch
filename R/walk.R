#' AST Walker for Torch Expression Analysis
#'
#' Walks an R expression AST and classifies nodes to identify
#' torch operations vs graph-breaking R code.

# Known torch tensor methods that can stay in a graph
.torch_methods <- c(

  # Arithmetic
  "add", "sub", "mul", "div", "matmul", "mm", "bmm",
  "pow", "sqrt", "exp", "log", "log2", "log10",
  "abs", "neg", "sign", "floor", "ceil", "round", "trunc",
  # Comparison

  "eq", "ne", "lt", "le", "gt", "ge",
  # Reduction
"sum", "mean", "prod", "max", "min", "argmax", "argmin",
  "all", "any",
  # Shape
  "view", "reshape", "squeeze", "unsqueeze", "flatten",
  "transpose", "permute", "contiguous", "expand", "repeat",
  "narrow", "slice", "chunk", "split",
  # Activation
  "relu", "sigmoid", "tanh", "softmax", "log_softmax",
  "gelu", "silu", "leaky_relu", "elu",
  # Normalization
  "layer_norm", "batch_norm",
  # Linear algebra
  "inverse", "det", "svd", "qr", "cholesky",
  # Indexing (these are tricky - may cause graph breaks)
  "index_select", "gather", "scatter",
  # Type/device
  "to", "float", "double", "int", "long", "bool",
  "cpu", "cuda", "contiguous",
  # Cloning
  "clone", "detach"
)

# Known torch_* functions
.torch_functions <- c(
  "torch_tensor", "torch_zeros", "torch_ones", "torch_randn", "torch_rand",
  "torch_empty", "torch_full", "torch_arange", "torch_linspace",
  "torch_eye", "torch_diag",
  "torch_cat", "torch_stack", "torch_concat",
  "torch_sum", "torch_mean", "torch_prod", "torch_max", "torch_min",
  "torch_matmul", "torch_mm", "torch_bmm",
  "torch_relu", "torch_sigmoid", "torch_tanh", "torch_softmax",
  "torch_exp", "torch_log", "torch_sqrt", "torch_pow", "torch_abs",
  "torch_clamp", "torch_where",
  "torch_einsum",
  "torch_linear", "torch_layer_norm",
  "torch_gelu", "torch_silu", "torch_sigmoid",
  "torch_conv1d"
)

# Operators that work on tensors
.tensor_operators <- c("+", "-", "*", "/", "^", "%%", "%/%", "%*%",
                       "==", "!=", "<", "<=", ">", ">=",
                       "&", "|", "!")

# Known side-effect functions (cause graph breaks)
.side_effect_functions <- c(
  "print", "cat", "message", "warning", "stop",
  "writeLines", "write", "save", "saveRDS",
  "plot", "lines", "points", "abline"
)

#' Classify an AST Node
#'
#' @param expr An R expression (call, symbol, or literal)
#' @return A list with `type` and additional metadata
#' @noRd
classify_node <- function(expr) {
  # Literals
  if (is.numeric(expr) || is.character(expr) || is.logical(expr)) {
    return(list(type = "literal", value = expr))
  }

  # Symbols (variable names)
  if (is.symbol(expr)) {
    name <- as.character(expr)
    return(list(type = "symbol", name = name))
  }

  # Not a call - unknown
  if (!is.call(expr)) {
    return(list(type = "unknown", expr = expr))
  }

  fn <- expr[[1]]

  # Block expression { ... }
  if (identical(fn, as.symbol("{"))) {
    return(list(type = "block", n_statements = length(expr) - 1L))
  }

  # Assignment <- or =
  if (identical(fn, as.symbol("<-")) || identical(fn, as.symbol("="))) {
    target <- expr[[2]]
    value <- expr[[3]]
    return(list(
      type = "assignment",
      target = if (is.symbol(target)) as.character(target) else deparse(target),
      value_expr = value
    ))
  }

  # Control flow
  if (identical(fn, as.symbol("if"))) {
    return(list(type = "if", has_else = length(expr) == 4L))
  }
  if (identical(fn, as.symbol("for"))) {
    return(list(type = "for"))
  }
  if (identical(fn, as.symbol("while"))) {
    return(list(type = "while"))
  }

  # Method call: x$method(args)
  if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
    method_name <- as.character(fn[[3]])
    object_expr <- fn[[2]]
    args <- if (length(expr) > 1) as.list(expr[-1]) else list()

    is_torch_method <- method_name %in% .torch_methods

    return(list(
      type = "method_call",
      method = method_name,
      object_expr = object_expr,
      args = args,
      is_torch_method = is_torch_method,
      graph_safe = is_torch_method
    ))
  }

  # $ accessor without call (just x$foo, not x$foo())
  if (identical(fn, as.symbol("$"))) {
    return(list(
      type = "accessor",
      object_expr = expr[[2]],
      field = as.character(expr[[3]])
    ))
  }

  # Function call
  if (is.symbol(fn)) {
    fn_name <- as.character(fn)

    # Binary/unary operators
    if (fn_name %in% .tensor_operators) {
      return(list(
        type = "operator",
        op = fn_name,
        args = as.list(expr[-1]),
        graph_safe = TRUE  # Assuming tensor operands
      ))
    }

    # torch_* functions
    if (fn_name %in% .torch_functions) {
      return(list(
        type = "torch_function",
        fn = fn_name,
        args = as.list(expr[-1]),
        graph_safe = TRUE
      ))
    }

    # Side effect functions
    if (fn_name %in% .side_effect_functions) {
      return(list(
        type = "side_effect",
        fn = fn_name,
        args = as.list(expr[-1]),
        graph_safe = FALSE
      ))
    }

    # Unknown function call
    return(list(
      type = "function_call",
      fn = fn_name,
      args = as.list(expr[-1]),
      graph_safe = FALSE  # Conservative: unknown = graph break
    ))
  }

  # Complex function expression (e.g., (function(x) x)(y))
  list(type = "complex_call", expr = expr, graph_safe = FALSE)
}


#' Walk an Expression and Build a Node Tree
#'
#' Recursively walks an R expression, classifying each node.
#'
#' @param expr An R expression
#' @param depth Current recursion depth (for debugging)
#' @return A nested list representing the classified AST
#' @examples
#' walked <- walk_expr(quote(y <- x$matmul(w)))
#' walked$type
#' @export
walk_expr <- function(expr, depth = 0L) {
  node <- classify_node(expr)
  node$depth <- depth

  # Recursively walk children based on node type
  if (node$type == "block") {
    node$statements <- lapply(
      as.list(expr[-1]),  # Skip the `{`
      walk_expr,
      depth = depth + 1L
    )
  } else if (node$type == "assignment") {
    node$value <- walk_expr(node$value_expr, depth + 1L)
    node$value_expr <- NULL  # Clean up
  } else if (node$type == "method_call") {
    node$object <- walk_expr(node$object_expr, depth + 1L)
    node$object_expr <- NULL
    node$args <- lapply(node$args, walk_expr, depth = depth + 1L)
  } else if (node$type == "operator") {
    node$operands <- lapply(node$args, walk_expr, depth = depth + 1L)
    node$args <- NULL
  } else if (node$type %in% c("torch_function", "function_call", "side_effect")) {
    node$arguments <- lapply(node$args, walk_expr, depth = depth + 1L)
    node$args <- NULL
  } else if (node$type == "accessor") {
    node$object <- walk_expr(node$object_expr, depth + 1L)
    node$object_expr <- NULL
  } else if (node$type == "if") {
    node$condition <- walk_expr(expr[[2]], depth + 1L)
    node$then_branch <- walk_expr(expr[[3]], depth + 1L)
    if (node$has_else) {
      node$else_branch <- walk_expr(expr[[4]], depth + 1L)
    }
  } else if (node$type == "for") {
    node$var <- as.character(expr[[2]])
    node$iterator <- walk_expr(expr[[3]], depth + 1L)
    node$body <- walk_expr(expr[[4]], depth + 1L)
  } else if (node$type == "while") {
    node$condition <- walk_expr(expr[[2]], depth + 1L)
    node$body <- walk_expr(expr[[3]], depth + 1L)
  }

  node
}


#' Check if an Expression is Graph-Safe
#'
#' Walks the expression tree and checks if all nodes are graph-safe
#' (can be compiled to a torch graph without breaks).
#'
#' @param expr An R expression or a walked node tree
#' @return Logical, TRUE if entirely graph-safe
#' @examples
#' is_graph_safe(quote(y <- x$matmul(w)))
#' is_graph_safe(quote(print(x)))
#' @export
is_graph_safe <- function(expr) {
  # If it's a raw expression, walk it first
  if (!is.list(expr) || is.null(expr$type)) {
    expr <- walk_expr(expr)
  }

  # Literals and symbols are always safe
  if (expr$type %in% c("literal", "symbol")) {
    return(TRUE)
  }

  # Check explicit graph_safe flag
  if (!is.null(expr$graph_safe) && !expr$graph_safe) {
    return(FALSE)
  }

  # Control flow is not graph-safe (data-dependent)
  if (expr$type %in% c("if", "for", "while")) {
    return(FALSE)
  }

  # Recursively check children
  children_safe <- TRUE

  if (!is.null(expr$statements)) {
    children_safe <- children_safe && all(vapply(expr$statements, is_graph_safe, logical(1)))
  }
  if (!is.null(expr$value)) {
    children_safe <- children_safe && is_graph_safe(expr$value)
  }
  if (!is.null(expr$object)) {
    children_safe <- children_safe && is_graph_safe(expr$object)
  }
  if (!is.null(expr$args) && is.list(expr$args)) {
    children_safe <- children_safe && all(vapply(expr$args, is_graph_safe, logical(1)))
  }
  if (!is.null(expr$operands)) {
    children_safe <- children_safe && all(vapply(expr$operands, is_graph_safe, logical(1)))
  }
  if (!is.null(expr$arguments)) {
    children_safe <- children_safe && all(vapply(expr$arguments, is_graph_safe, logical(1)))
  }

  children_safe
}


#' Find Graph Breaks in an Expression
#'
#' Identifies nodes that cause graph breaks.
#'
#' @param expr An R expression
#' @return A list of graph-breaking nodes with their details
#' @examples
#' find_graph_breaks(quote({ y <- x$relu(); print(y) }))
#' @export
find_graph_breaks <- function(expr) {
  if (!is.list(expr) || is.null(expr$type)) {
    expr <- walk_expr(expr)
  }

  breaks <- list()

  collect_breaks <- function(node, path = "") {
    # Check if this node breaks the graph
    if (!is.null(node$graph_safe) && !node$graph_safe) {
      breaks <<- c(breaks, list(list(
        type = node$type,
        detail = node$fn %||% node$method %||% node$type,
        path = path
      )))
    }

    # Control flow always breaks
    if (node$type %in% c("if", "for", "while")) {
      breaks <<- c(breaks, list(list(
        type = "control_flow",
        detail = node$type,
        path = path
      )))
    }

    # Recurse into children
    if (!is.null(node$statements)) {
      for (i in seq_along(node$statements)) {
        collect_breaks(node$statements[[i]], paste0(path, "/stmt", i))
      }
    }
    if (!is.null(node$value) && is.list(node$value) && !is.null(node$value$type)) {
      collect_breaks(node$value, paste0(path, "/value"))
    }
    if (!is.null(node$object)) {
      collect_breaks(node$object, paste0(path, "/object"))
    }
    if (!is.null(node$operands)) {
      for (i in seq_along(node$operands)) {
        collect_breaks(node$operands[[i]], paste0(path, "/operand", i))
      }
    }
    if (!is.null(node$arguments)) {
      for (i in seq_along(node$arguments)) {
        collect_breaks(node$arguments[[i]], paste0(path, "/arg", i))
      }
    }
    if (!is.null(node$condition)) {
      collect_breaks(node$condition, paste0(path, "/condition"))
    }
    if (!is.null(node$then_branch)) {
      collect_breaks(node$then_branch, paste0(path, "/then"))
    }
    if (!is.null(node$else_branch)) {
      collect_breaks(node$else_branch, paste0(path, "/else"))
    }
    if (!is.null(node$body)) {
      collect_breaks(node$body, paste0(path, "/body"))
    }
  }

  collect_breaks(expr)
  breaks
}

# Helper for NULL coalescing
`%||%` <- function(a, b) if (is.null(a)) b else a


#' Check if a Call is to an nn_module
#'
#' Examines a function call expression to determine if the function
#' being called is an nn_module (has a forward() method that can be
#' expanded via body()).
#'
#' @param expr A call expression
#' @param env Environment to resolve the callable in
#' @return Logical
#' @noRd
is_module_call <- function(expr, env = parent.frame()) {
  if (!is.call(expr)) return(FALSE)
  fn_expr <- expr[[1]]

  # Try to resolve what's being called
  obj <- tryCatch(eval(fn_expr, envir = env), error = function(e) NULL)
  if (is.null(obj)) return(FALSE)

  # Check if it's an nn_module callable
  is.function(obj) && inherits(obj, "nn_module")
}
