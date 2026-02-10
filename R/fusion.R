#' Kernel Fusion via Pattern Matching
#'
#' Detects fuseable operation patterns in expressions and dispatches
#' to pre-compiled SIMD kernels.

# Pattern definitions: list of (pattern, kernel_fn, description)
# Patterns are matched against method chains
.fusion_patterns <- list(
  # Method chains (most specific first)
  list(
    pattern = c("relu", "sigmoid", "tanh"),
    kernel = "fused_relu_sigmoid_tanh",
    description = "relu + sigmoid + tanh"
  ),
  list(
    pattern = c("relu", "sigmoid"),
    kernel = "fused_relu_sigmoid",
    description = "relu + sigmoid"
  ),

  # Single activations that are actually fusions
  # (these catch x * sigmoid(x) patterns rewritten as method chains)
  list(
    pattern = c("silu"),
    kernel = "fused_silu",
    description = "SiLU activation"
  ),
  list(
    pattern = c("gelu"),
    kernel = "fused_gelu",
    description = "GELU activation"
  )
)


#' Extract Method Chain from Expression
#'
#' Walks a method chain like x$relu()$sigmoid() and extracts the
#' base variable and operation sequence.
#'
#' @param expr An R expression
#' @return List with `base` (symbol or NULL), `ops` (character vector),
#'   and `args` (list of argument lists for each op)
#' @noRd
extract_method_chain <- function(expr) {
  ops <- character()
  args_list <- list()
  current <- expr

  # Walk the chain backwards

  while (is.call(current)) {
    fn <- current[[1]]

    # Method call: something$method(args)
    if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
      method <- as.character(fn[[3]])
      ops <- c(method, ops)  # prepend

      # Collect arguments (skip the first which is the call itself)
      call_args <- as.list(current)[-1]
      args_list <- c(list(call_args), args_list)

      current <- fn[[2]]  # the object
    } else {
      break
    }
  }

  # current should now be the base variable
  base <- if (is.symbol(current)) current else NULL

  list(base = base, ops = ops, args = args_list)
}


#' Check if Expression Matches a SiLU Pattern
#'
#' SiLU is x * sigmoid(x), which appears as:
#' - x * torch_sigmoid(x)
#' - x * x$sigmoid()
#' - torch_mul(x, torch_sigmoid(x))
#'
#' @param expr An R expression
#' @param env Environment to check variable identity
#' @return List with `match` (logical), `input` (symbol if matched)
#' @noRd
match_silu_pattern <- function(expr, env = NULL) {
  if (!is.call(expr)) return(list(match = FALSE))

  fn <- expr[[1]]

  # Pattern: x * x$sigmoid()
  if (identical(fn, as.symbol("*")) && length(expr) == 3) {
    lhs <- expr[[2]]
    rhs <- expr[[3]]

    # Check if rhs is lhs$sigmoid()
    if (is.call(rhs)) {
      chain <- extract_method_chain(rhs)
      if (length(chain$ops) == 1 &&
          chain$ops[1] == "sigmoid" &&
          is.symbol(chain$base) &&
          is.symbol(lhs) &&
          identical(chain$base, lhs)) {
        return(list(match = TRUE, input = lhs))
      }
    }

    # Check reverse: x$sigmoid() * x
    if (is.call(lhs)) {
      chain <- extract_method_chain(lhs)
      if (length(chain$ops) == 1 &&
          chain$ops[1] == "sigmoid" &&
          is.symbol(chain$base) &&
          is.symbol(rhs) &&
          identical(chain$base, rhs)) {
        return(list(match = TRUE, input = rhs))
      }
    }
  }

  # Pattern: x * torch_sigmoid(x)
  if (identical(fn, as.symbol("*")) && length(expr) == 3) {
    lhs <- expr[[2]]
    rhs <- expr[[3]]

    if (is.call(rhs) && length(rhs) >= 2) {
      rhs_fn <- rhs[[1]]
      if ((is.symbol(rhs_fn) && as.character(rhs_fn) == "torch_sigmoid") ||
          (is.call(rhs_fn) && identical(rhs_fn[[1]], as.symbol("::")) &&
           as.character(rhs_fn[[3]]) == "torch_sigmoid")) {
        rhs_arg <- rhs[[2]]
        if (is.symbol(lhs) && is.symbol(rhs_arg) && identical(lhs, rhs_arg)) {
          return(list(match = TRUE, input = lhs))
        }
      }
    }
  }

  list(match = FALSE)
}


#' Match Expression Against Fusion Patterns
#'
#' @param expr An R expression (single statement)
#' @param env Environment for variable lookup
#' @return List with `matched` (logical), `kernel` (function name),
#'   `input` (symbol), `description` (string)
#' @noRd
match_fusion_pattern <- function(expr, env = NULL) {
  # First check special patterns (like SiLU)
  silu <- match_silu_pattern(expr, env)
  if (silu$match) {
    return(list(
      matched = TRUE,
      kernel = "fused_silu",
      input = silu$input,
      description = "SiLU: x * sigmoid(x)"
    ))
  }

  # Check method chains
  chain <- extract_method_chain(expr)

  if (is.null(chain$base) || length(chain$ops) == 0) {
    return(list(matched = FALSE))
  }

  # Try to match against known patterns
  for (pat in .fusion_patterns) {
    if (identical(chain$ops, pat$pattern)) {
      # Check that all ops have no extra arguments
      # (fused kernels assume default args)
      all_empty <- all(vapply(chain$args, function(a) length(a) == 0, logical(1)))
      if (all_empty) {
        return(list(
          matched = TRUE,
          kernel = pat$kernel,
          input = chain$base,
          description = pat$description
        ))
      }
    }
  }

  list(matched = FALSE)
}


#' Execute Statement with Fusion
#'
#' Tries to execute a statement using fused kernels. Falls back to
#' regular evaluation if no pattern matches.
#'
#' @param stmt An R expression (single statement)
#' @param env Evaluation environment
#' @param verbose Print fusion info
#' @return List with `result`, `fused` (logical)
#' @noRd
execute_with_fusion <- function(stmt, env, verbose = FALSE) {
  # Handle assignment: extract RHS for pattern matching
  is_assignment <- is.call(stmt) &&
    (identical(stmt[[1]], as.symbol("<-")) || identical(stmt[[1]], as.symbol("=")))

  if (is_assignment) {
    lhs <- stmt[[2]]
    rhs <- stmt[[3]]
    target_var <- if (is.symbol(lhs)) as.character(lhs) else NULL
  } else {
    rhs <- stmt
    target_var <- NULL
  }

  # Try to match pattern

  match <- match_fusion_pattern(rhs, env)

  if (match$matched) {
    # Get input tensor
    input_name <- as.character(match$input)
    input_tensor <- tryCatch(
      get(input_name, envir = env, inherits = TRUE),
      error = function(e) NULL
    )

    if (!is.null(input_tensor) && inherits(input_tensor, "torch_tensor")) {
      # Get the kernel function
      kernel_fn <- tryCatch(
        get(match$kernel, envir = asNamespace("Rtorch")),
        error = function(e) NULL
      )

      if (!is.null(kernel_fn)) {
        if (verbose) {
          cat(sprintf("    FUSED: %s -> %s(%s)\n",
                      match$description, match$kernel, input_name))
        }

        # Execute fused kernel
        result <- kernel_fn(input_tensor)

        # Assign if needed
        if (!is.null(target_var)) {
          assign(target_var, result, envir = env)
        }

        return(list(result = result, fused = TRUE))
      }
    }
  }

  # No fusion - regular execution
  result <- eval(stmt, envir = env)
  list(result = result, fused = FALSE)
}


#' Check if Segment Can Be Fused
#'
#' Analyzes a segment to determine if it contains fuseable patterns.
#'
#' @param statements List of statements in the segment
#' @param env Evaluation environment
#' @return List with `can_fuse` (logical), `patterns` (list of matches)
#' @noRd
analyze_fusion_opportunities <- function(statements, env = NULL) {
  patterns <- list()

  for (i in seq_along(statements)) {
    stmt <- statements[[i]]

    # Handle assignment
    if (is.call(stmt) &&
        (identical(stmt[[1]], as.symbol("<-")) || identical(stmt[[1]], as.symbol("=")))) {
      rhs <- stmt[[3]]
    } else {
      rhs <- stmt
    }

    match <- match_fusion_pattern(rhs, env)
    if (match$matched) {
      patterns[[length(patterns) + 1]] <- list(
        index = i,
        match = match
      )
    }
  }

  list(
    can_fuse = length(patterns) > 0,
    patterns = patterns
  )
}
