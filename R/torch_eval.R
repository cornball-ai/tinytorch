#' Evaluate R Expressions as Torch Computations
#'
#' Captures an R expression and evaluates it, optionally compiling
#' graph-safe segments using jit_trace or fused SIMD kernels.
#'
#' @param expr An R expression involving torch tensor operations.
#'   Captured unevaluated via NSE.
#' @param ... Named torch tensors to make available in the expression.
#' @param .env Environment for resolving non-tensor symbols.
#' @param .compile Logical. If TRUE, attempts to compile graph-safe
#'   segments using jit_trace. Default FALSE (direct execution).
#' @param .fuse Logical. If TRUE, detects fuseable patterns and
#'   dispatches to pre-compiled SIMD kernels. Default TRUE.
#' @param .verbose Logical. If TRUE, prints compilation info.
#'
#' @return The result of evaluating the expression (typically a torch tensor).
#'
#' @examples
#' \dontrun{
#' x <- torch_randn(3, 3)
#' y <- torch_randn(3, 3)
#'
#' # Direct execution with automatic fusion (default)
#' result <- torch_eval({
#'   z <- x$matmul(y)
#'   z$relu()$sigmoid()
#' }, x = x, y = y)
#'
#' # With jit_trace compilation
#' result <- torch_eval({
#'   z <- x$matmul(y)
#'   z$relu()
#' }, x = x, y = y, .compile = TRUE)
#' }
#'
#' @export
torch_eval <- function(expr, ..., .env = parent.frame(), .compile = FALSE,
                       .fuse = TRUE, .verbose = FALSE) {
  # Capture the unevaluated expression
  captured <- substitute(expr)

  # Collect named tensor arguments
  tensors <- list(...)

  # Validate tensor arguments
  if (length(tensors) > 0) {
    if (is.null(names(tensors)) || any(names(tensors) == "")) {
      stop("All tensor arguments must be named", call. = FALSE)
    }
    for (nm in names(tensors)) {
      if (!inherits(tensors[[nm]], "torch_tensor")) {
        stop(sprintf("Argument '%s' must be a torch_tensor", nm), call. = FALSE)
      }
    }
  }

  # Build evaluation environment with tensors
  eval_env <- new.env(parent = .env)
  for (nm in names(tensors)) {
    eval_env[[nm]] <- tensors[[nm]]
  }

  if (.compile || .fuse) {
    # Use compiled execution with segmentation and/or fusion
    result <- execute_compiled(captured, eval_env,
                               compile = .compile,
                               fuse = .fuse,
                               verbose = .verbose)
  } else {
    # Direct evaluation
    result <- eval(captured, envir = eval_env)
  }

  result
}
