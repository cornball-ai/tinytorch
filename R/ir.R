#' Intermediate Representation (IR) for Torch Computation Graphs
#'
#' Provides a proper IR that replaces raw R AST as the working
#' representation for optimization passes.

#' Create an IR Node
#'
#' @param id Unique integer ID
#' @param op Operation name ("matmul", "relu", "add", "constant", "input", ...)
#' @param inputs Integer vector of input node IDs
#' @param attrs Named list of attributes (dtype, shape, value, name, etc.)
#' @return An ir_node list
#' @noRd
ir_node <- function(id, op, inputs = integer(), attrs = list()) {
  structure(
    list(
      id = as.integer(id),
      op = op,
      inputs = as.integer(inputs),
      attrs = attrs
    ),
    class = "ir_node"
  )
}


#' Create an IR Graph
#'
#' @param nodes Named list of ir_node, keyed by ID (as character)
#' @param input_ids Integer vector of input node IDs
#' @param output_ids Integer vector of output node IDs
#' @return An ir_graph list
#' @noRd
ir_graph <- function(nodes, input_ids, output_ids) {
  structure(
    list(
      nodes = nodes,
      input_ids = as.integer(input_ids),
      output_ids = as.integer(output_ids)
    ),
    class = "ir_graph"
  )
}


#' Lower R AST Statements to IR
#'
#' Takes a graph-safe segment's statements (raw R expressions) plus
#' the evaluation environment. Walks each statement recursively,
#' emitting IR nodes.
#'
#' @param statements List of R expressions (from segment_expr())
#' @param env Environment with tensor values
#' @return An ir_graph
#' @examples
#' \dontrun{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env()
#' e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' print(g)
#' }
#' @export
lower_to_ir <- function(statements, env = parent.frame()) {
  # State for the lowering pass
  next_id <- 1L
  nodes <- list()
  var_map <- new.env(parent = emptyenv())  # symbol name -> node ID
  input_ids <- integer()

  new_id <- function() {
    id <- next_id
    next_id <<- next_id + 1L
    id
  }

  add_node <- function(node) {
    nodes[[as.character(node$id)]] <<- node
    node$id
  }

  # Create or retrieve an input node for a symbol
  get_or_create_input <- function(name) {
    if (exists(name, envir = var_map, inherits = FALSE)) {
      return(get(name, envir = var_map, inherits = FALSE))
    }
    id <- new_id()
    node <- ir_node(id, "input", attrs = list(name = name))
    add_node(node)
    assign(name, id, envir = var_map)
    input_ids <<- c(input_ids, id)
    id
  }

  # Recursive expression lowering
  lower_expr <- function(expr) {

    # Literal values
    if (is.numeric(expr) || is.logical(expr)) {
      id <- new_id()
      return(add_node(ir_node(id, "constant", attrs = list(value = expr))))
    }

    if (is.character(expr)) {
      id <- new_id()
      return(add_node(ir_node(id, "constant", attrs = list(value = expr))))
    }

    # Symbol lookup
    if (is.symbol(expr)) {
      name <- as.character(expr)
      return(get_or_create_input(name))
    }

    if (!is.call(expr)) {
      # Unknown expression type - treat as opaque constant
      id <- new_id()
      return(add_node(ir_node(id, "constant", attrs = list(value = expr))))
    }

    fn <- expr[[1]]

    # Block expression { ... }
    if (identical(fn, as.symbol("{"))) {
      last_id <- NULL
      for (i in seq_along(expr)[-1]) {
        last_id <- lower_expr(expr[[i]])
      }
      return(last_id)
    }

    # Assignment: z <- expr
    if (identical(fn, as.symbol("<-")) || identical(fn, as.symbol("="))) {
      target <- expr[[2]]
      rhs <- expr[[3]]
      rhs_id <- lower_expr(rhs)
      if (is.symbol(target)) {
        assign(as.character(target), rhs_id, envir = var_map)
      }
      return(rhs_id)
    }

    # Method call: obj$method(args...)
    if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
      method_name <- as.character(fn[[3]])
      obj_expr <- fn[[2]]

      obj_id <- lower_expr(obj_expr)
      arg_ids <- integer()
      if (length(expr) > 1) {
        for (i in seq_along(expr)[-1]) {
          arg_ids <- c(arg_ids, lower_expr(expr[[i]]))
        }
      }

      id <- new_id()
      return(add_node(ir_node(id, method_name, inputs = c(obj_id, arg_ids))))
    }

    # $ accessor without call (x$field)
    if (identical(fn, as.symbol("$"))) {
      obj_id <- lower_expr(expr[[2]])
      field <- as.character(expr[[3]])
      id <- new_id()
      return(add_node(ir_node(id, "getattr", inputs = obj_id,
                              attrs = list(field = field))))
    }

    # Handle namespace-qualified calls: pkg::fn(args) → treat as fn(args)
    if (is.call(fn) && identical(fn[[1]], as.symbol("::")) && length(fn) == 3L) {
      fn <- fn[[3]]  # Extract the function name
      expr[[1]] <- fn  # Replace in expr so arg processing works
    }

    if (is.symbol(fn)) {
      fn_name <- as.character(fn)

      # Parentheses: (expr) - just pass through
      if (fn_name == "(") {
        return(lower_expr(expr[[2]]))
      }

      # Unary negation: -x
      if (fn_name == "-" && length(expr) == 2L) {
        operand_id <- lower_expr(expr[[2]])
        id <- new_id()
        return(add_node(ir_node(id, "neg", inputs = operand_id)))
      }

      # Unary logical not: !x
      if (fn_name == "!" && length(expr) == 2L) {
        operand_id <- lower_expr(expr[[2]])
        id <- new_id()
        return(add_node(ir_node(id, "not", inputs = operand_id)))
      }

      # Binary/unary operators
      op_map <- c(
        "+" = "add", "-" = "sub", "*" = "mul", "/" = "div",
        "^" = "pow", "%%" = "remainder", "%/%" = "floor_div",
        "%*%" = "matmul",
        "==" = "eq", "!=" = "ne", "<" = "lt", "<=" = "le",
        ">" = "gt", ">=" = "ge",
        "&" = "and", "|" = "or"
      )

      if (fn_name %in% names(op_map)) {
        lhs_id <- lower_expr(expr[[2]])
        rhs_id <- lower_expr(expr[[3]])
        id <- new_id()
        return(add_node(ir_node(id, op_map[[fn_name]],
                                inputs = c(lhs_id, rhs_id))))
      }

      # torch_* functions
      if (fn_name %in% .torch_functions) {
        arg_ids <- integer()
        arg_attrs <- list()
        if (length(expr) > 1) {
          nms <- names(expr)
          for (i in seq_along(expr)[-1]) {
            arg_val <- expr[[i]]
            nm <- if (!is.null(nms) && nms[i] != "") nms[i] else NULL

            # Literal args go to attrs, tensor args to inputs
            if (is.numeric(arg_val) || is.integer(arg_val)) {
              if (!is.null(nm)) {
                arg_attrs[[nm]] <- arg_val
              } else {
                arg_attrs[[paste0("arg", i - 1L)]] <- arg_val
              }
            } else {
              arg_ids <- c(arg_ids, lower_expr(arg_val))
            }
          }
        }
        id <- new_id()
        attrs <- c(list(fn = fn_name), arg_attrs)
        return(add_node(ir_node(id, fn_name, inputs = arg_ids, attrs = attrs)))
      }

      # Other known function - lower args
      arg_ids <- integer()
      arg_names <- character()
      if (length(expr) > 1) {
        nms <- names(expr)
        for (i in seq_along(expr)[-1]) {
          arg_ids <- c(arg_ids, lower_expr(expr[[i]]))
          nm <- if (!is.null(nms) && nms[i] != "") nms[i] else ""
          arg_names <- c(arg_names, nm)
        }
      }
      id <- new_id()
      fn_attrs <- list(fn = fn_name)
      # Preserve argument names for list() calls
      if (fn_name == "list" && any(arg_names != "")) {
        fn_attrs$names <- arg_names
      }
      return(add_node(ir_node(id, fn_name, inputs = arg_ids,
                              attrs = fn_attrs)))
    }

    # Fallback: opaque expression
    id <- new_id()
    add_node(ir_node(id, "opaque", attrs = list(expr = expr)))
  }

  # Lower each statement, track the last result
  last_id <- NULL
  for (stmt in statements) {
    last_id <- lower_expr(stmt)
  }

  # Output is the last expression's node
  output_ids <- if (!is.null(last_id)) last_id else integer()

  ir_graph(nodes, input_ids, output_ids)
}


#' Format an IR Node as a String
#'
#' @param node An ir_node
#' @return Character string representation
#' @noRd
format_ir_node <- function(node) {
  id_str <- sprintf("%%%d", node$id)

  # Build type suffix if shape info is available
  type_suffix <- ""
  if (!is.null(node$attrs$output_dtype) && !is.null(node$attrs$output_shape)) {
    shape_str <- paste(node$attrs$output_shape, collapse = ", ")
    type_suffix <- sprintf(" : %s[%s]", node$attrs$output_dtype, shape_str)
  }

  if (node$op == "input") {
    name <- node$attrs$name %||% "?"
    return(sprintf("%s = input[%s]%s", id_str, name, type_suffix))
  }

  if (node$op == "constant") {
    val <- node$attrs$value
    val_str <- if (is.numeric(val) && length(val) == 1) {
      as.character(val)
    } else if (is.character(val) && length(val) == 1) {
      sprintf("\"%s\"", val)
    } else {
      deparse(val, width.cutoff = 60)[1]
    }
    return(sprintf("%s = constant(%s)%s", id_str, val_str, type_suffix))
  }

  if (length(node$inputs) > 0) {
    input_strs <- paste0("%", node$inputs, collapse = ", ")
    sprintf("%s = %s(%s)%s", id_str, node$op, input_strs, type_suffix)
  } else {
    sprintf("%s = %s()%s", id_str, node$op, type_suffix)
  }
}


#' Print an IR Graph
#'
#' @param x An ir_graph
#' @param ... Ignored
#' @return Invisibly returns x
#' @examples
#' \dontrun{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env()
#' e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' print(g)
#' }
#' @export
print.ir_graph <- function(x, ...) {
  # Print nodes in ID order
  ids <- sort(as.integer(names(x$nodes)))
  for (id in ids) {
    node <- x$nodes[[as.character(id)]]
    cat(format_ir_node(node), "\n")
  }
  if (length(x$output_ids) > 0) {
    out_strs <- paste0("%", x$output_ids, collapse = ", ")
    cat(sprintf("return %s\n", out_strs))
  }
  invisible(x)
}


#' Validate an IR Graph
#'
#' Checks that all input references resolve, there are no cycles,
#' and nodes are in topological order.
#'
#' @param graph An ir_graph
#' @return TRUE if valid, otherwise stops with an error
#' @examples
#' \dontrun{
#' stmts <- list(quote(y <- x$relu()))
#' e <- new.env()
#' e$x <- torch_randn(c(2, 3))
#' g <- lower_to_ir(stmts, e)
#' validate_ir(g)
#' }
#' @export
validate_ir <- function(graph) {
  if (!inherits(graph, "ir_graph")) {
    stop("Expected an ir_graph object", call. = FALSE)
  }

  node_ids <- as.integer(names(graph$nodes))

  # Check all input references resolve

  for (id_str in names(graph$nodes)) {
    node <- graph$nodes[[id_str]]
    for (inp in node$inputs) {
      if (!as.character(inp) %in% names(graph$nodes)) {
        stop(sprintf("Node %%%d references non-existent input %%%d",
                     node$id, inp), call. = FALSE)
      }
    }
  }

  # Check output_ids reference existing nodes
  for (out_id in graph$output_ids) {
    if (!as.character(out_id) %in% names(graph$nodes)) {
      stop(sprintf("Output %%%d references non-existent node", out_id),
           call. = FALSE)
    }
  }

  # Check input_ids reference existing nodes
  for (inp_id in graph$input_ids) {
    if (!as.character(inp_id) %in% names(graph$nodes)) {
      stop(sprintf("Input ID %%%d references non-existent node", inp_id),
           call. = FALSE)
    }
  }

  # Check topological order: every node's inputs must have lower IDs
  for (id_str in names(graph$nodes)) {
    node <- graph$nodes[[id_str]]
    for (inp in node$inputs) {
      if (inp >= node$id) {
        stop(sprintf("Node %%%d references input %%%d (not in topological order)",
                     node$id, inp), call. = FALSE)
      }
    }
  }

  # Check no duplicate IDs
  if (anyDuplicated(node_ids)) {
    stop("Duplicate node IDs found", call. = FALSE)
  }

  TRUE
}
