#' Compilation Layer for torchlang
#'
#' Compiles graph-safe segments using jit_trace, with fallback to
#' direct execution for R code segments.

#' Analyze Variable Flow in an Expression
#'
#' Identifies which variables are read (inputs) and written (outputs)
#' in an expression. Used to determine function signatures for compilation.
#'
#' @param expr An R expression
#' @return A list with `reads` and `writes` character vectors
#' @export
analyze_variables <- function(expr) {
  reads <- character()
  writes <- character()

  collect_vars <- function(e, in_lhs = FALSE) {
    if (is.symbol(e)) {
      name <- as.character(e)
      # Skip special symbols
      if (name %in% c("{", "<-", "=", "$", "if", "for", "while", "function")) {
        return()
      }
      if (in_lhs) {
        writes <<- c(writes, name)
      } else {
        reads <<- c(reads, name)
      }
      return()
    }

    if (!is.call(e)) return()

    fn <- e[[1]]

    # Block - recurse into statements
    if (identical(fn, as.symbol("{"))) {
      for (i in seq_along(e)[-1]) {
        collect_vars(e[[i]], in_lhs = FALSE)
      }
      return()
    }

    # Assignment - LHS is write, RHS is read
    if (identical(fn, as.symbol("<-")) || identical(fn, as.symbol("="))) {
      lhs <- e[[2]]
      rhs <- e[[3]]
      # Simple assignment: x <- ...
      if (is.symbol(lhs)) {
        writes <<- c(writes, as.character(lhs))
      }
      # Could be x$field <- ... (ignore for now)
      collect_vars(rhs, in_lhs = FALSE)
      return()
    }

    # Method call: x$method(args) - x is read, args are read
    if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
      collect_vars(fn[[2]], in_lhs = FALSE)  # object
      for (i in seq_along(e)[-1]) {
        collect_vars(e[[i]], in_lhs = FALSE)  # args
      }
      return()
    }

    # $ accessor: x$field - x is read
    if (identical(fn, as.symbol("$"))) {
      collect_vars(e[[2]], in_lhs = FALSE)
      return()
    }

    # Regular function call - all args are reads
    for (i in seq_along(e)[-1]) {
      collect_vars(e[[i]], in_lhs = FALSE)
    }
  }

  collect_vars(expr)

  # Remove duplicates, sort for consistency
  reads <- sort(unique(reads))
  writes <- sort(unique(writes))

  # Inputs = reads that weren't written first (in this expression)
  # This is a simplification - proper analysis would track order
  inputs <- setdiff(reads, writes)

  list(
    reads = reads,
    writes = writes,
    inputs = inputs,
    outputs = writes
  )
}


#' Wrap Statements in a Function
#'
#' Creates a function from a list of statements with specified inputs.
#'
#' @param statements List of R expressions
#' @param inputs Character vector of input variable names
#' @return A function
#' @noRd
wrap_in_function <- function(statements, inputs) {
  # Build function body
  if (length(statements) == 1) {
    body_expr <- statements[[1]]
  } else {
    body_expr <- as.call(c(as.symbol("{"), statements))
  }

  # Build argument list
  args <- replicate(length(inputs), substitute(), simplify = FALSE)
  names(args) <- inputs

  # Create function
  f <- eval(call("function", as.pairlist(args), body_expr))

  f
}


#' Compile a Graph Segment
#'
#' Compiles a graph-safe segment using jit_trace with caching.
#' Non-tensor values are captured in a closure (become constants in traced graph).
#' Cache key includes expression structure and tensor shapes.
#'
#' @param statements List of R expressions (the segment)
#' @param env Environment with tensor values for tracing
#' @return A list with `fn` (traced function), `inputs`, `outputs`
#' @noRd
compile_segment <- function(statements, env) {
  # Combine statements for analysis
  if (length(statements) == 1) {
    combined <- statements[[1]]
  } else {
    combined <- as.call(c(as.symbol("{"), statements))
  }

  # Analyze variables
  vars <- analyze_variables(combined)

  # Separate tensor inputs from non-tensor values
  tensor_inputs <- character()
  nontensor_values <- list()

  for (nm in vars$inputs) {
    val <- tryCatch(
      get(nm, envir = env, inherits = TRUE),
      error = function(e) NULL
    )
    if (is.null(val)) {
      # Variable not found - will fail at runtime, skip
      next
    }
    if (inherits(val, "torch_tensor")) {
      tensor_inputs <- c(tensor_inputs, nm)
    } else {
      nontensor_values[[nm]] <- val
    }
  }

  # Get tensor values for tracing
  tensor_values <- lapply(tensor_inputs, get, envir = env, inherits = TRUE)
  names(tensor_values) <- tensor_inputs

  # Build cache key from expression structure + tensor shapes
  expr_key <- paste(deparse(combined), collapse = "\n")
  tensor_shapes <- lapply(tensor_values, function(t) as.integer(t$shape))
  cache_key <- make_cache_key(expr_key, tensor_shapes)

  # Check cache first
  cached <- cache_get(cache_key)
  if (!is.null(cached)) {
    return(list(
      traced_fn = cached,
      wrapper_fn = NULL,  # Not needed when using cached
      tensor_inputs = tensor_inputs,
      nontensor_inputs = names(nontensor_values),
      outputs = vars$outputs,
      cache_hit = TRUE
    ))
  }

  # Create wrapper function with closure over non-tensor values
  # The function only takes tensor args; non-tensors are captured
  closure_env <- list2env(nontensor_values, parent = env)
  wrapper_fn <- wrap_in_function(statements, tensor_inputs)
  environment(wrapper_fn) <- closure_env

  # Trace it - only tensor inputs are passed (positional, not named)
  traced_fn <- tryCatch({
    if (length(tensor_inputs) > 0) {
      do.call(jit_trace, c(list(wrapper_fn), unname(tensor_values)))
    } else {
      # No tensor inputs - can't trace, will execute directly
      NULL
    }
  }, error = function(e) {
    warning(sprintf("jit_trace failed: %s. Falling back to direct execution.",
                    conditionMessage(e)), call. = FALSE)
    NULL
  })

  # Cache the result (traced_fn if available, otherwise wrapper_fn)
  fn_to_cache <- if (!is.null(traced_fn)) traced_fn else wrapper_fn
  if (!is.null(fn_to_cache)) {
    cache_set(cache_key, fn_to_cache)
  }

  list(
    traced_fn = traced_fn,
    wrapper_fn = wrapper_fn,
    tensor_inputs = tensor_inputs,
    nontensor_inputs = names(nontensor_values),
    outputs = vars$outputs,
    cache_hit = FALSE
  )
}


#' Execute Expression with Compilation
#'
#' Segments an expression, compiles graph-safe segments, and executes.
#' Supports two optimization strategies:
#' - **Fusion** (.fuse): Pattern-matches operations and dispatches to
#'   pre-compiled SIMD kernels. Fast, no compilation overhead.
#' - **Compilation** (.compile): Traces segments with jit_trace.
#'   Better for long chains but has trace overhead.
#'
#' Fusion is tried first. Operations that don't match a fused pattern
#' fall through to jit_trace (if enabled) or direct execution.
#'
#' @param expr An R expression
#' @param env Environment with tensors
#' @param compile Logical, whether to use jit_trace
#' @param fuse Logical, whether to try fused kernel dispatch
#' @param verbose Logical, whether to print compilation info
#' @return The result of the expression
#' @export
execute_compiled <- function(expr, env, compile = TRUE, fuse = TRUE,
                             verbose = FALSE) {
  segments <- segment_expr(expr)

  if (verbose) {
    cat(sprintf("Executing %d segment(s), fuse=%s, compile=%s\n",
                length(segments), fuse, compile))
  }

  # Create mutable environment for execution
  exec_env <- new.env(parent = env)

  result <- NULL

  for (i in seq_along(segments)) {
    seg <- segments[[i]]

    if (seg$type == "graph") {

      if (verbose) cat(sprintf("  Segment %d [GRAPH]: %d statement(s)\n",
                               i, length(seg$statements)))

      # Build IR for this segment (does not change execution yet)
      ir <- tryCatch(
        lower_to_ir(seg$statements, exec_env),
        error = function(e) NULL
      )
      if (verbose && !is.null(ir)) {
        # Try shape inference if inputs are available
        ir_shapes <- list()
        for (inp_id in ir$input_ids) {
          nm <- ir$nodes[[as.character(inp_id)]]$attrs$name
          val <- tryCatch(get(nm, envir = exec_env, inherits = TRUE),
                          error = function(e) NULL)
          if (inherits(val, "torch_tensor")) {
            ir_shapes[[nm]] <- as.integer(val$shape)
          }
        }
        if (length(ir_shapes) > 0) {
          ir <- tryCatch(infer_shapes(ir, ir_shapes), error = function(e) ir)
        }
        cat("  IR:\n")
        ir_lines <- capture.output(print(ir))
        for (ln in ir_lines) cat("    ", ln, "\n")
      }

      # Process statements: try fusion first, then compile, then eval
      fused_all <- TRUE

      for (j in seq_along(seg$statements)) {
        stmt <- seg$statements[[j]]

        # Try fusion first
        if (fuse) {
          fused <- execute_with_fusion(stmt, exec_env, verbose = verbose)
          if (fused$fused) {
            result <- fused$result
            next
          }
        }

        fused_all <- FALSE

        # If not fused and compile is on, compile the remaining segment
        if (compile && j == 1) {
          # Only try jit_trace for the whole segment on first unfused statement
          compiled <- compile_segment(seg$statements, exec_env)

          if (!is.null(compiled$traced_fn)) {
            input_values <- lapply(compiled$tensor_inputs, get, envir = exec_env)
            result <- do.call(compiled$traced_fn, input_values)

            if (length(compiled$outputs) == 1 && !is.null(result)) {
              assign(compiled$outputs[1], result, envir = exec_env)
            }

            if (verbose) {
              cache_status <- if (isTRUE(compiled$cache_hit)) "CACHED" else "traced"
              cat(sprintf("    COMPILED: %s, %d tensor inputs\n",
                          cache_status, length(compiled$tensor_inputs)))
            }
            break  # Whole segment handled by jit_trace
          }
        }

        # Fallback: direct execution
        if (verbose) cat(sprintf("    EVAL: statement %d\n", j))
        result <- eval(stmt, envir = exec_env)
      }

    } else {
      # R code segment - direct execution
      if (verbose) cat(sprintf("  Segment %d [R]: direct execution\n", i))
      for (stmt in seg$statements) {
        result <- eval(stmt, envir = exec_env)
      }
    }
  }

  result
}
