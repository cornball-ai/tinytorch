#' nn_module Tracing via Recursive body() Expansion
#'
#' Captures nn_module forward() bodies as ASTs, recursively expands
#' sub-module calls, resolves parameters, and produces flat tensor
#' op sequences that feed into the existing IR pipeline.

# nnf_ -> torch_* mapping for common functions
# n_args: how many positional args to pass to torch_fn (drop inplace etc.)
# torch_args: named arg mapping for torch_fn (when names differ)
.nnf_registry <- list(
  nnf_linear     = list(torch_fn = "torch_linear",
                         args = c("input", "weight", "bias"),
                         n_args = 3L),
  nnf_relu       = list(torch_fn = "torch_relu",
                         args = c("input", "inplace"),
                         n_args = 1L),
  nnf_gelu       = list(torch_fn = "torch_gelu",
                         args = c("self", "approximate"),
                         n_args = 1L,
                         torch_args = c("self")),
  nnf_silu       = list(torch_fn = "torch_silu",
                         args = c("input", "inplace"),
                         n_args = 1L),
  nnf_sigmoid    = list(torch_fn = "torch_sigmoid",
                         args = c("input"),
                         n_args = 1L),
  nnf_tanh       = list(torch_fn = "torch_tanh",
                         args = c("input"),
                         n_args = 1L),
  nnf_softmax    = list(torch_fn = "torch_softmax",
                         args = c("self", "dim", "dtype"),
                         n_args = 3L,
                         torch_args = c("self", "dim", "dtype")),
  nnf_log_softmax = list(torch_fn = "torch_log_softmax",
                          args = c("self", "dim", "dtype"),
                          n_args = 3L,
                          torch_args = c("self", "dim", "dtype")),
  nnf_layer_norm = list(torch_fn = "torch_layer_norm",
                         args = c("input", "normalized_shape", "weight",
                                  "bias", "eps"),
                         n_args = 5L),
  nnf_batch_norm = list(torch_fn = "torch_batch_norm",
                         args = c("input", "weight", "bias",
                                  "running_mean", "running_var",
                                  "training", "momentum", "eps"),
                         n_args = 8L),
  nnf_dropout    = list(torch_fn = NULL,
                         args = c("input", "p", "training")),
  nnf_embedding  = list(torch_fn = "torch_embedding",
                         args = c("weight", "indices"),
                         n_args = 2L),
  nnf_conv1d     = list(torch_fn = "torch_conv1d",
                         args = c("input", "weight", "bias", "stride",
                                  "padding", "dilation", "groups"),
                         n_args = 7L)
)


#' Check if an Object is an nn_module Callable
#'
#' @param obj Any R object
#' @return Logical
#' @noRd
is_nn_module_callable <- function(obj) {
  is.function(obj) && inherits(obj, "nn_module")
}


#' Get the R6 Module Instance from a Callable
#'
#' nn_module callables carry a reference to their R6 instance via
#' the environment of the forward method.
#'
#' @param callable An nn_module callable
#' @return The R6 nn_Module instance, or NULL
#' @noRd
get_module_instance <- function(callable) {
  # The forward method's environment contains self
  fwd <- callable$forward
  if (is.null(fwd)) return(NULL)
  env <- environment(fwd)
  if (is.null(env)) return(NULL)
  if (exists("self", envir = env, inherits = FALSE)) {
    get("self", envir = env, inherits = FALSE)
  } else {
    NULL
  }
}


#' Get Private Environment of a Module
#'
#' @param instance R6 nn_Module instance
#' @return The private environment
#' @noRd
get_private <- function(instance) {
  instance$.__enclos_env__$private
}


#' Resolve a self$field Access
#'
#' Given a field name and module instance, determines what the field is
#' (parameter, buffer, sub-module, scalar, method) and returns
#' the appropriate representation.
#'
#' @param field_name Character, the field being accessed
#' @param instance R6 nn_Module instance
#' @param prefix Character, dot-separated parameter name prefix
#' @return A list with type and value/symbol
#' @noRd
resolve_self_access <- function(field_name, instance, prefix = "") {
  priv <- get_private(instance)

  # Check parameters
  if (field_name %in% names(priv$parameters_)) {
    param_name <- if (nzchar(prefix)) {
      paste0(prefix, ".", field_name)
    } else {
      field_name
    }
    return(list(
      type = "parameter",
      name = param_name,
      value = priv$parameters_[[field_name]]
    ))
  }

  # Check buffers
  if (field_name %in% names(priv$buffers_)) {
    buf_name <- if (nzchar(prefix)) {
      paste0(prefix, ".", field_name)
    } else {
      field_name
    }
    return(list(
      type = "buffer",
      name = buf_name,
      value = priv$buffers_[[field_name]]
    ))
  }

  # Check sub-modules
  if (field_name %in% names(priv$modules_)) {
    return(list(
      type = "module",
      name = field_name,
      value = priv$modules_[[field_name]]
    ))
  }

  # Check scalar public fields (training, eps, normalized_shape, etc.)
  # Use sentinel to distinguish "field errors" from "field is NULL"
  .missing <- structure(list(), class = ".resolve_missing")
  val <- tryCatch(instance[[field_name]], error = function(e) .missing)
  if (inherits(val, ".resolve_missing")) {
    return(list(type = "unknown", name = field_name))
  }
  if (is.null(val)) {
    # NULL field (e.g., nn_linear bias when bias=FALSE)
    return(list(type = "scalar", name = field_name, value = NULL))
  }
  if (inherits(val, "torch_tensor")) {
    # Could be a parameter accessed via public interface
    param_name <- if (nzchar(prefix)) {
      paste0(prefix, ".", field_name)
    } else {
      field_name
    }
    return(list(type = "parameter", name = param_name, value = val))
  }
  if (is.function(val)) {
    return(list(type = "method", name = field_name, value = val))
  }
  # Scalar value (numeric, logical, integer, character, list)
  list(type = "scalar", name = field_name, value = val)
}


#' Expand an nnf_ Call to torch_* Operations
#'
#' @param fn_name Character, e.g. "nnf_linear"
#' @param arg_exprs List of argument expressions (already rewritten)
#' @param arg_names Character vector of argument names (may have "")
#' @return Rewritten expression, or NULL if cannot expand
#' @noRd
expand_nnf <- function(fn_name, arg_exprs, arg_names = NULL) {
  entry <- .nnf_registry[[fn_name]]
  if (!is.null(entry)) {
    if (is.null(entry$torch_fn)) {
      # Special cases like dropout
      if (fn_name == "nnf_dropout") {
        # In eval mode, dropout is identity
        # Check if training arg is FALSE
        training_idx <- which(entry$args == "training")
        if (length(training_idx) > 0 && length(arg_exprs) >= training_idx) {
          training_val <- arg_exprs[[training_idx]]
          if (is.logical(training_val) && !training_val) {
            return(arg_exprs[[1]])  # identity: return input
          }
        }
      }
      return(NULL)  # graph break
    }

    # Only pass the first n_args arguments to torch_fn
    # (drops inplace, training, etc. that the torch_* function doesn't accept)
    n <- entry$n_args %||% length(arg_exprs)
    call_args <- arg_exprs[seq_len(min(n, length(arg_exprs)))]

    # Use torch_args names if specified, otherwise pass unnamed
    if (!is.null(entry$torch_args)) {
      names(call_args) <- entry$torch_args[seq_len(length(call_args))]
    }
    return(as.call(c(as.symbol(entry$torch_fn), call_args)))
  }

  # Not in registry — try to get the function body
  fn <- tryCatch(getFromNamespace(fn_name, "Rtorch"), error = function(e) NULL)
  if (is.null(fn) || !is.function(fn)) return(NULL)

  fn_body <- body(fn)
  if (is.null(fn_body)) return(NULL)

  # Simple case: body is { torch_foo(...) } — single torch_* call
  stmts <- if (is.call(fn_body) && identical(fn_body[[1]], as.symbol("{"))) {
    as.list(fn_body[-1])
  } else {
    list(fn_body)
  }

  # Check if it's a single torch_* call (skip if/else bodies)
  if (length(stmts) == 1 && is.call(stmts[[1]])) {
    inner <- stmts[[1]]
    inner_fn <- if (is.symbol(inner[[1]])) as.character(inner[[1]]) else ""
    if (grepl("^torch_", inner_fn)) {
      # Substitute formal args with our args
      fmls <- names(formals(fn))
      sub_env <- new.env(parent = emptyenv())
      for (i in seq_along(fmls)) {
        if (i <= length(arg_exprs)) {
          assign(fmls[i], arg_exprs[[i]], envir = sub_env)
        }
      }
      return(do.call(substitute, list(inner, sub_env)))
    }
  }

  NULL
}


#' Check if an Expression is Statically Resolvable
#'
#' Tries to evaluate a condition expression that may reference scalars
#' known at trace time.
#'
#' @param expr An R expression (condition)
#' @param known_values Named list of known scalar values
#' @return The logical value if resolvable, or NULL if dynamic
#' @noRd
try_resolve_static <- function(expr, known_values) {
  tryCatch({
    if (is.environment(known_values)) {
      env <- known_values
    } else {
      env <- list2env(known_values, parent = baseenv())
    }
    val <- eval(expr, envir = env)
    if (is.logical(val) && length(val) == 1) val
    else NULL
  }, error = function(e) NULL)
}


#' Expand a Module's forward() Body
#'
#' Recursively rewrites the AST of a module's forward method:
#' - self$param -> unique parameter symbol
#' - self$submodule(args) -> recursive expansion
#' - nnf_foo(args) -> torch_* call
#' - static if/else -> resolved branch
#' - nn_sequential for loop -> unrolled
#'
#' @param module An nn_module callable or R6 instance
#' @param arg_exprs Named list of argument expressions
#' @param prefix Dot-separated name prefix for parameters
#' @param depth Current recursion depth (to prevent infinite loops)
#' @param max_depth Maximum recursion depth
#' @return A list with: statements (list of rewritten exprs),
#'   params (named list of parameter tensors),
#'   graph_breaks (list of break descriptions)
#' @examples
#' \donttest{
#' m <- nn_linear(3, 2)
#' expand_module(m, arg_exprs = list(input = as.symbol("input")))
#' }
#' @export
expand_module <- function(module, arg_exprs = list(), prefix = "",
                          depth = 0L, max_depth = 10L) {
  if (depth > max_depth) {
    return(list(
      statements = list(),
      params = list(),
      graph_breaks = list(list(
        reason = "max recursion depth exceeded",
        prefix = prefix, depth = depth
      ))
    ))
  }

  # Get instance and forward body
  if (is_nn_module_callable(module)) {
    instance <- get_module_instance(module)
    fwd <- module$forward
  } else if (inherits(module, "nn_Module") || inherits(module, "nn_module")) {
    instance <- module
    fwd <- module$forward
  } else {
    return(list(
      statements = list(),
      params = list(),
      graph_breaks = list(list(
        reason = sprintf("not an nn_module: %s", paste(class(module), collapse = ",")),
        prefix = prefix, depth = depth
      ))
    ))
  }

  if (is.null(instance) || is.null(fwd)) {
    return(list(
      statements = list(),
      params = list(),
      graph_breaks = list(list(reason = "no forward method", prefix = prefix))
    ))
  }

  fwd_body <- body(fwd)
  fwd_formals <- names(formals(fwd))

  # Build known values: map formal names to arg_exprs
  formal_map <- new.env(parent = emptyenv())
  for (i in seq_along(fwd_formals)) {
    fname <- fwd_formals[i]
    if (fname %in% names(arg_exprs)) {
      assign(fname, arg_exprs[[fname]], envir = formal_map)
    } else if (i <= length(arg_exprs) && is.null(names(arg_exprs))) {
      assign(fname, arg_exprs[[i]], envir = formal_map)
    }
  }

  # Collect parameters and graph breaks
  params <- list()
  graph_breaks <- list()

  # Known scalar values for static resolution (training mode, etc.)
  # Use environment (not list) so NULL bindings are preserved.
  # list(x = NULL) drops the element; env$x <- NULL keeps it.
  known_scalars <- new.env(parent = baseenv())
  # Add default values of formals — including NULL defaults for
  # unprovided args. This lets !is.null(x) resolve to FALSE when
  # x has default NULL and wasn't passed by the caller.
  fwd_defaults <- formals(fwd)
  for (nm in names(fwd_defaults)) {
    # Safely check if this formal has a default (not missing)
    has_default <- tryCatch({
      d <- fwd_defaults[[nm]]
      !is.symbol(d) || nzchar(as.character(d))
    }, error = function(e) FALSE)
    if (has_default && !nm %in% names(arg_exprs)) {
      val <- tryCatch(fwd_defaults[[nm]], error = function(e) NULL)
      if (is.null(val) || (!is.symbol(val) && !is.language(val))) {
        assign(nm, val, envir = known_scalars)
        # Also put in formal_map so the default substitutes in expressions
        assign(nm, val, envir = formal_map)
      }
    }
  }

  # Check if this is nn_sequential (special case: for loop)
  priv <- get_private(instance)
  is_sequential <- "nn_sequential" %in% class(module) ||
    (inherits(instance, "nn_Module") &&
       any(grepl("nn_sequential", class(instance), fixed = TRUE)))

  # Only unroll actual nn_sequential modules (not any module with a for loop)
  if (is_sequential) {
    return(unroll_sequential(instance, arg_exprs, prefix, depth, max_depth))
  }

  # Rewrite the body
  rewrite <- function(expr) {
    # Literals pass through
    if (is.numeric(expr) || is.logical(expr) || is.character(expr)) {
      return(expr)
    }

    # Symbol: substitute formal args
    if (is.symbol(expr)) {
      if (identical(expr, quote(expr=))) return(expr)  # missing arg (trailing comma)
      name <- as.character(expr)
      if (exists(name, envir = formal_map, inherits = FALSE)) {
        return(get(name, envir = formal_map, inherits = FALSE))
      }
      return(expr)
    }

    if (!is.call(expr)) return(expr)

    fn <- expr[[1]]

    # Block: rewrite each statement
    if (identical(fn, as.symbol("{"))) {
      rewritten <- list(as.symbol("{"))
      for (i in seq_along(expr)[-1]) {
        rewritten <- c(rewritten, list(rewrite(expr[[i]])))
      }
      return(as.call(rewritten))
    }

    # Assignment: rewrite RHS, preserve target
    if (identical(fn, as.symbol("<-")) || identical(fn, as.symbol("="))) {
      target <- expr[[2]]
      rhs <- rewrite(expr[[3]])
      # Track local variable assignments in formal_map
      if (is.symbol(target)) {
        assign(as.character(target), target, envir = formal_map)
      }
      return(call("<-", target, rhs))
    }

    # self$field access (no call)
    if (identical(fn, as.symbol("$")) && is.symbol(expr[[2]]) &&
        as.character(expr[[2]]) == "self") {
      field_name <- as.character(expr[[3]])
      resolved <- resolve_self_access(field_name, instance, prefix)

      if (resolved$type == "parameter" || resolved$type == "buffer") {
        sym <- as.symbol(resolved$name)
        params[[resolved$name]] <<- resolved$value
        return(sym)
      }
      if (resolved$type == "scalar") {
        # Inline scalar value
        return(resolved$value)
      }
      if (resolved$type == "module") {
        # Bare module reference (not called) — return symbol
        return(as.symbol(paste0(prefix, ".", field_name)))
      }
      # Unknown self access → graph break
      graph_breaks[[length(graph_breaks) + 1L]] <<- list(
        reason = sprintf("unknown self$%s", field_name),
        prefix = prefix
      )
      return(expr)
    }

    # self$method(args) or self$submodule(args)
    if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
      obj <- fn[[2]]
      method_name <- as.character(fn[[3]])

      # self$submodule(args) or self$method(args)
      if (is.symbol(obj) && as.character(obj) == "self") {
        resolved <- resolve_self_access(method_name, instance, prefix)

        if (resolved$type == "module") {
          # Recursive expansion of sub-module
          sub_module <- resolved$value
          sub_prefix <- if (nzchar(prefix)) {
            paste0(prefix, ".", method_name)
          } else {
            method_name
          }

          # Rewrite args first
          sub_args <- list()
          sub_arg_names <- names(expr)
          for (i in seq_along(expr)[-1]) {
            arg <- rewrite(expr[[i]])
            nm <- if (!is.null(sub_arg_names) && nzchar(sub_arg_names[i])) {
              sub_arg_names[i]
            } else {
              NULL
            }
            sub_args <- c(sub_args, list(arg))
            if (!is.null(nm)) names(sub_args)[length(sub_args)] <- nm
          }

          sub_result <- expand_module(sub_module, sub_args,
                                       prefix = sub_prefix,
                                       depth = depth + 1L,
                                       max_depth = max_depth)
          params <<- c(params, sub_result$params)
          graph_breaks <<- c(graph_breaks, sub_result$graph_breaks)

          # If sub-module expanded cleanly, inline its statements
          if (length(sub_result$graph_breaks) == 0 &&
              length(sub_result$statements) > 0) {
            return(inline_statements(sub_result$statements))
          }

          # Graph break — leave as opaque call with rewritten args
          graph_breaks[[length(graph_breaks) + 1L]] <<- list(
            reason = sprintf("sub-module %s had graph breaks", method_name),
            prefix = prefix
          )
          new_call <- as.call(c(list(fn), sub_args))
          return(new_call)
        }

        if (resolved$type == "method") {
          # Self method call — inline the body
          method_fn <- resolved$value
          method_body <- body(method_fn)
          if (!is.null(method_body)) {
            method_formals <- names(formals(method_fn))
            method_args <- list()
            for (i in seq_along(expr)[-1]) {
              method_args <- c(method_args, list(rewrite(expr[[i]])))
            }
            # Save and extend formal_map with method's formals
            saved_bindings <- list()
            for (j in seq_along(method_formals)) {
              fname <- method_formals[j]
              if (exists(fname, envir = formal_map, inherits = FALSE)) {
                saved_bindings[[fname]] <- get(fname, envir = formal_map,
                                               inherits = FALSE)
              }
              if (j <= length(method_args)) {
                assign(fname, method_args[[j]], envir = formal_map)
              }
            }
            # Rewrite the method body (self refs still resolve correctly)
            inlined <- rewrite(method_body)
            # Restore formal_map
            for (fname in method_formals) {
              if (fname %in% names(saved_bindings)) {
                assign(fname, saved_bindings[[fname]], envir = formal_map)
              } else {
                if (exists(fname, envir = formal_map, inherits = FALSE)) {
                  rm(list = fname, envir = formal_map)
                }
              }
            }
            return(inlined)
          }
          graph_breaks[[length(graph_breaks) + 1L]] <<- list(
            reason = sprintf("self method: self$%s()", method_name),
            prefix = prefix
          )
          return(expr)
        }

        if (resolved$type == "parameter" || resolved$type == "buffer") {
          # self$weight(args) — calling a tensor (shouldn't happen normally)
          # More likely self$weight used in method position — rewrite obj
          params[[resolved$name]] <<- resolved$value
          new_fn <- call("$", as.symbol(resolved$name), fn[[3]])
          rewritten_args <- lapply(seq_along(expr)[-1], function(i) {
            rewrite(expr[[i]])
          })
          return(as.call(c(list(new_fn), rewritten_args)))
        }
      }

      # self$module$forward(args) — explicit $forward call on sub-module
      if (is.call(obj) && identical(obj[[1]], as.symbol("$")) &&
          is.symbol(obj[[2]]) && as.character(obj[[2]]) == "self" &&
          method_name == "forward") {
        module_field <- as.character(obj[[3]])
        resolved <- resolve_self_access(module_field, instance, prefix)

        if (resolved$type == "module") {
          sub_module <- resolved$value
          sub_prefix <- if (nzchar(prefix)) {
            paste0(prefix, ".", module_field)
          } else {
            module_field
          }

          sub_args <- list()
          sub_arg_names <- names(expr)
          for (i in seq_along(expr)[-1]) {
            arg <- rewrite(expr[[i]])
            nm <- if (!is.null(sub_arg_names) && nzchar(sub_arg_names[i])) {
              sub_arg_names[i]
            } else {
              NULL
            }
            sub_args <- c(sub_args, list(arg))
            if (!is.null(nm)) names(sub_args)[length(sub_args)] <- nm
          }

          sub_result <- expand_module(sub_module, sub_args,
                                       prefix = sub_prefix,
                                       depth = depth + 1L,
                                       max_depth = max_depth)
          params <<- c(params, sub_result$params)
          graph_breaks <<- c(graph_breaks, sub_result$graph_breaks)

          if (length(sub_result$graph_breaks) == 0 &&
              length(sub_result$statements) > 0) {
            return(inline_statements(sub_result$statements))
          }

          graph_breaks[[length(graph_breaks) + 1L]] <<- list(
            reason = sprintf("sub-module %s had graph breaks", module_field),
            prefix = prefix
          )
          new_call <- as.call(c(list(fn), sub_args))
          return(new_call)
        }
      }

      # Not self — could be tensor method call: x$matmul(y)
      # Rewrite object and args
      new_obj <- rewrite(obj)
      new_fn <- call("$", new_obj, fn[[3]])
      rewritten_args <- lapply(seq_along(expr)[-1], function(i) {
        rewrite(expr[[i]])
      })
      return(as.call(c(list(new_fn), rewritten_args)))
    }

    # $ accessor on non-self: rewrite object
    if (identical(fn, as.symbol("$"))) {
      new_obj <- rewrite(expr[[2]])
      return(call("$", new_obj, expr[[3]]))
    }

    # nnf_* function calls
    if (is.symbol(fn)) {
      fn_name <- as.character(fn)

      if (grepl("^nnf_", fn_name)) {
        # Rewrite args first
        rewritten_args <- list()
        arg_names <- character()
        expr_names <- names(expr)
        for (i in seq_along(expr)[-1]) {
          rewritten_args <- c(rewritten_args, list(rewrite(expr[[i]])))
          nm <- if (!is.null(expr_names) && nzchar(expr_names[i])) {
            expr_names[i]
          } else {
            ""
          }
          arg_names <- c(arg_names, nm)
        }

        expanded <- expand_nnf(fn_name, rewritten_args, arg_names)
        if (!is.null(expanded)) return(expanded)

        # Could not expand — graph break
        graph_breaks[[length(graph_breaks) + 1L]] <<- list(
          reason = sprintf("unexpandable nnf_: %s", fn_name),
          prefix = prefix
        )
        return(as.call(c(as.symbol(fn_name), rewritten_args)))
      }

      # torch_* functions — pass through with rewritten args
      if (fn_name %in% .torch_functions || grepl("^torch_", fn_name)) {
        rewritten_args <- lapply(seq_along(expr)[-1], function(i) {
          rewrite(expr[[i]])
        })
        nms <- names(expr)
        if (!is.null(nms)) names(rewritten_args) <- nms[-1]
        return(as.call(c(as.symbol(fn_name), rewritten_args)))
      }

      # Side effects → graph break
      if (fn_name %in% .side_effect_functions) {
        graph_breaks[[length(graph_breaks) + 1L]] <<- list(
          reason = sprintf("side effect: %s()", fn_name),
          prefix = prefix
        )
        return(expr)
      }

      # size/shape/dim → graph break (data-dependent)
      if (fn_name %in% c("dim", "length", "nrow", "ncol")) {
        graph_breaks[[length(graph_breaks) + 1L]] <<- list(
          reason = sprintf("data-dependent: %s()", fn_name),
          prefix = prefix
        )
        return(expr)
      }
    }

    # if/else: try static resolution
    if (identical(fn, as.symbol("if"))) {
      cond <- expr[[2]]
      rewritten_cond <- rewrite(cond)

      # Try to resolve statically
      resolved <- try_resolve_static(rewritten_cond, known_scalars)
      if (!is.null(resolved)) {
        if (isTRUE(resolved)) {
          return(rewrite(expr[[3]]))  # take then-branch
        } else if (length(expr) == 4L) {
          return(rewrite(expr[[4]]))  # take else-branch
        } else {
          return(NULL)  # no else, condition false → nothing
        }
      }

      # Dynamic condition → graph break
      graph_breaks[[length(graph_breaks) + 1L]] <<- list(
        reason = sprintf("dynamic if: %s", deparse(cond, width.cutoff = 60)[1]),
        prefix = prefix
      )
      # Rewrite branches anyway (partial tracing)
      then_branch <- rewrite(expr[[3]])
      if (length(expr) == 4L) {
        else_branch <- rewrite(expr[[4]])
        return(call("if", rewritten_cond, then_branch, else_branch))
      }
      return(call("if", rewritten_cond, then_branch))
    }

    # for loop → graph break (unless it's sequential pattern, handled above)
    if (identical(fn, as.symbol("for"))) {
      graph_breaks[[length(graph_breaks) + 1L]] <<- list(
        reason = "for loop",
        prefix = prefix
      )
      return(expr)
    }

    # return() → rewrite the value
    if (identical(fn, as.symbol("return"))) {
      if (length(expr) > 1) {
        return(rewrite(expr[[2]]))
      }
      return(NULL)
    }

    # Generic function call: rewrite all args
    rewritten_args <- lapply(seq_along(expr)[-1], function(i) {
      rewrite(expr[[i]])
    })
    nms <- names(expr)
    if (!is.null(nms)) names(rewritten_args) <- nms[-1]
    as.call(c(list(fn), rewritten_args))
  }

  # Run the rewriter
  result_expr <- rewrite(fwd_body)

  # Extract statements from block
  statements <- if (is.call(result_expr) &&
                    identical(result_expr[[1]], as.symbol("{"))) {
    as.list(result_expr[-1])
  } else if (!is.null(result_expr)) {
    list(result_expr)
  } else {
    list()
  }

  # Filter NULL statements (from resolved-away branches)
  statements <- Filter(Negate(is.null), statements)

  list(
    statements = statements,
    params = params,
    graph_breaks = graph_breaks
  )
}


#' Check if a Function Body Matches nn_sequential Pattern
#'
#' @param body_expr A body expression
#' @return Logical
#' @noRd
is_sequential_body <- function(body_expr) {
  if (!is.call(body_expr)) return(FALSE)
  # Look for: { for (module in ...) { input <- module(input) } ; input }
  stmts <- if (identical(body_expr[[1]], as.symbol("{"))) {
    as.list(body_expr[-1])
  } else {
    list(body_expr)
  }
  for (s in stmts) {
    if (is.call(s) && identical(s[[1]], as.symbol("for"))) {
      return(TRUE)
    }
  }
  FALSE
}


#' Unroll an nn_sequential Module
#'
#' @param instance R6 module instance
#' @param arg_exprs Named list with input expression
#' @param prefix Name prefix
#' @param depth Current depth
#' @param max_depth Maximum depth
#' @return Same structure as expand_module return
#' @noRd
unroll_sequential <- function(instance, arg_exprs, prefix, depth, max_depth) {
  priv <- get_private(instance)
  sub_modules <- priv$modules_

  if (length(sub_modules) == 0) {
    return(list(statements = list(), params = list(), graph_breaks = list()))
  }

  # The input is the first argument
  # nn_sequential$forward signature is forward(input)
  if (length(arg_exprs) > 0) {
    current_var <- if (!is.null(names(arg_exprs)) && "input" %in% names(arg_exprs)) {
      arg_exprs[["input"]]
    } else {
      arg_exprs[[1]]
    }
  } else {
    current_var <- as.symbol("input")
  }

  all_statements <- list()
  all_params <- list()
  all_breaks <- list()

  for (nm in names(sub_modules)) {
    sub_mod <- sub_modules[[nm]]
    sub_prefix <- if (nzchar(prefix)) paste0(prefix, ".", nm) else nm

    # Expand each sub-module
    sub_result <- expand_module(
      sub_mod,
      arg_exprs = list(input = current_var),
      prefix = sub_prefix,
      depth = depth + 1L,
      max_depth = max_depth
    )

    all_params <- c(all_params, sub_result$params)
    all_breaks <- c(all_breaks, sub_result$graph_breaks)

    if (length(sub_result$statements) > 0 &&
        length(sub_result$graph_breaks) == 0) {
      # Clean expansion — create a temp variable for each layer
      temp_var <- as.symbol(sprintf(".seq_%s", gsub("\\.", "_", sub_prefix)))

      # Wrap statements as: temp <- { expanded_stmts }
      if (length(sub_result$statements) == 1) {
        assign_expr <- call("<-", temp_var, sub_result$statements[[1]])
      } else {
        block <- as.call(c(as.symbol("{"), sub_result$statements))
        assign_expr <- call("<-", temp_var, block)
      }
      all_statements <- c(all_statements, list(assign_expr))
      current_var <- temp_var
    } else {
      # Graph break in sub-module — cannot inline
      all_breaks <- c(all_breaks, list(list(
        reason = sprintf("graph break in sequential layer %s", nm),
        prefix = sub_prefix
      )))
      # Still advance current_var to a temp
      temp_var <- as.symbol(sprintf(".seq_%s", gsub("\\.", "_", sub_prefix)))
      current_var <- temp_var
    }
  }

  # Final statement: return the last temp variable
  if (length(all_statements) > 0) {
    all_statements <- c(all_statements, list(current_var))
  }

  list(
    statements = all_statements,
    params = all_params,
    graph_breaks = all_breaks
  )
}


#' Inline Expanded Statements into a Single Expression
#'
#' When a sub-module expands to multiple statements, wraps them
#' in a block { ... } so they can be used as a single expression.
#'
#' @param statements List of expressions
#' @return A single expression
#' @noRd
inline_statements <- function(statements) {
  if (length(statements) == 0) return(NULL)
  if (length(statements) == 1) return(statements[[1]])
  as.call(c(as.symbol("{"), statements))
}


#' Trace an nn_module for Optimized Execution
#'
#' Expands a module's forward() body via recursive AST rewriting,
#' resolves parameters and sub-modules, and optionally optimizes
#' through the IR pipeline.
#'
#' @param module An nn_module callable
#' @param ... Named example tensors matching the forward() signature
#' @param .optimize Logical, run optimization passes on the IR
#' @param .fuse Logical, compile fusion groups to kernels
#' @param .backend Character: "auto", "gpu", or "cpu". "auto" detects
#'   from input tensors at call time and ariel availability.
#' @param .verbose Logical, print tracing info
#' @return A list with:
#'   \item{fn}{A function that executes the traced graph}
#'   \item{expanded}{The expanded statement list}
#'   \item{params}{Named list of parameter tensors}
#'   \item{graph_breaks}{List of graph break descriptions}
#'   \item{ir}{The IR graph (if no graph breaks), or NULL}
#' @examples
#' \donttest{
#' m <- nn_linear(3, 2)
#' traced <- trace_module(m, input = torch_randn(c(1, 3)))
#' traced$fn(input = torch_randn(c(1, 3)))
#' }
#' @export
trace_module <- function(module, ..., .optimize = TRUE, .fuse = FALSE,
                         .backend = "auto", .verbose = FALSE) {
  example_inputs <- list(...)
  if (length(example_inputs) == 0) {
    stop("At least one named example input is required", call. = FALSE)
  }
  if (is.null(names(example_inputs)) || any(names(example_inputs) == "")) {
    stop("All example inputs must be named", call. = FALSE)
  }

  # Convert example inputs to symbols for expansion
  arg_exprs <- lapply(names(example_inputs), as.symbol)
  names(arg_exprs) <- names(example_inputs)

  if (.verbose) {
    message("Expanding module forward()...")
  }

  # Expand the module
  expanded <- expand_module(module, arg_exprs = arg_exprs)

  if (.verbose) {
    message(sprintf("  %d statement(s), %d parameter(s), %d graph break(s)",
                    length(expanded$statements),
                    length(expanded$params),
                    length(expanded$graph_breaks)))
    if (length(expanded$graph_breaks) > 0) {
      message("  Graph breaks:")
      for (gb in expanded$graph_breaks) {
        message(sprintf("    - %s", gb$reason))
      }
    }
  }

  params <- expanded$params

  # Build the execution function
  if (length(expanded$graph_breaks) == 0 && length(expanded$statements) > 0) {
    # Clean trace — can build IR
    ir <- tryCatch({
      all_inputs <- c(names(example_inputs), names(params))
      g <- lower_to_ir(expanded$statements)

      if (.optimize) {
        g <- optimize_graph(g)
      }
      g
    }, error = function(e) {
      if (.verbose) message(sprintf("  IR lowering failed: %s", conditionMessage(e)))
      NULL
    })

    # Build execution function that uses the IR executor
    fn <- make_traced_fn(expanded$statements, params, names(example_inputs),
                         ir = ir, optimize = .optimize, fuse = .fuse,
                         backend = .backend, module = module)
  } else {
    # Has graph breaks — build segmented execution function
    ir <- NULL
    fn <- make_traced_fn(expanded$statements, params, names(example_inputs),
                         ir = NULL, optimize = FALSE, fuse = FALSE,
                         backend = .backend, module = module)
  }

  structure(
    list(
      fn = fn,
      expanded = expanded$statements,
      params = params,
      graph_breaks = expanded$graph_breaks,
      ir = ir
    ),
    class = "traced_module"
  )
}


#' Build a Traced Execution Function
#'
#' @param statements Expanded statement list
#' @param params Named list of parameter tensors
#' @param input_names Character vector of user input names
#' @param ir Optimized IR graph or NULL
#' @param optimize Whether to optimize
#' @param fuse Whether to fuse
#' @param backend "auto", "gpu", or "cpu"
#' @return A function
#' @noRd
make_traced_fn <- function(statements, params, input_names,
                           ir = NULL, optimize = TRUE, fuse = TRUE,
                           backend = "auto", module = NULL) {
  # Capture in closure
  force(statements)
  force(params)
  force(input_names)
  force(ir)
  force(optimize)
  force(fuse)
  force(backend)
  force(module)

  # Cache for GPU-migrated params (keyed by device string)
  .param_cache <- new.env(parent = emptyenv())

  # Pre-compute IR input names (constant)
  .ir_input_names <- if (!is.null(ir)) {
    vapply(ir$input_ids, function(id) {
      ir$nodes[[as.character(id)]]$attrs$name %||% ""
    }, character(1))
  }

  # Pre-compute graph fingerprint (constant — graph never changes)
  .graph_fp <- if (!is.null(ir)) .graph_fingerprint(ir)

  # Prepared graph cache (keyed by shape_fp||backend)
  .prep_cache <- new.env(parent = emptyenv())
  # Last shape fingerprint for fast-path comparison
  .last_shape_fp <- ""
  .last_prepared <- NULL

  function(...) {
    call_args <- list(...)
    if (is.null(names(call_args)) || any(names(call_args) == "")) {
      if (length(call_args) == length(input_names)) {
        names(call_args) <- input_names
      } else {
        stop("Arguments must be named or match the number of inputs",
             call. = FALSE)
      }
    }

    # Detect target device from inputs
    target_device <- NULL
    for (t in call_args) {
      if (inherits(t, "torch_tensor") && t$is_cuda) {
        target_device <- t$device
        break
      }
    }

    # Auto-migrate params to target device if needed
    active_params <- params
    if (!is.null(target_device)) {
      dev_key <- as.character(target_device)
      if (exists(dev_key, envir = .param_cache, inherits = FALSE)) {
        active_params <- get(dev_key, envir = .param_cache, inherits = FALSE)
      } else {
        migrated <- lapply(params, function(p) {
          if (inherits(p, "torch_tensor") && !p$is_cuda) {
            p$to(device = target_device)
          } else {
            p
          }
        })
        names(migrated) <- names(params)
        assign(dev_key, migrated, envir = .param_cache)
        active_params <- migrated
      }
    }

    # --- Fast IR execution path ---
    # Pre-computed graph_fp + C++ shape fingerprint + local prep cache
    # bypasses execute_optimized() entirely on cache hit.
    if (!is.null(ir)) {
      all_inputs <- c(call_args, active_params)
      inputs_for_ir <- all_inputs[.ir_input_names]

      # Resolve backend
      resolved_backend <- if (backend == "auto") {
        if (!is.null(target_device)) {
          if (requireNamespace("ariel", quietly = TRUE)) "gpu" else "cpu"
        } else {
          "cpu"
        }
      } else {
        backend
      }

      # Shape fingerprint via C++ (one .Call replaces 14 R->C++ crossings)
      shape_fp <- .shapes_fingerprint(inputs_for_ir)
      cache_key <- paste0(shape_fp, "||", resolved_backend)

      # Fast path: same shapes as last call → skip cache lookup entirely
      prepared <- NULL
      if (identical(cache_key, .last_shape_fp)) {
        prepared <- .last_prepared
      } else if (exists(cache_key, envir = .prep_cache, inherits = FALSE)) {
        prepared <- get(cache_key, envir = .prep_cache, inherits = FALSE)
      }

      if (is.null(prepared)) {
        # Cache miss: run full pipeline once
        prepared <- prepare_graph(ir, inputs_for_ir,
                                   optimize = optimize, fuse = fuse,
                                   backend = resolved_backend)
        assign(cache_key, prepared, envir = .prep_cache)
      }

      # Update last-call cache
      .last_shape_fp <<- cache_key
      .last_prepared <<- prepared

      # Direct fast_fn call (compiled executor, no per-node overhead)
      if (!is.null(prepared$fast_fn)) {
        result <- tryCatch(prepared$fast_fn(inputs_for_ir), error = function(e) NULL)
        if (!is.null(result)) return(result)
      }

      result <- tryCatch(
        execute_prepared(prepared, inputs_for_ir),
        error = function(e) NULL
      )
      if (!is.null(result)) return(result)
    }

    # Fallback: direct eval of expanded statements
    # Build execution environment with inputs + params
    exec_env <- new.env(parent = baseenv())
    for (nm in names(call_args)) exec_env[[nm]] <- call_args[[nm]]
    for (nm in names(active_params)) exec_env[[nm]] <- active_params[[nm]]

    if (!is.null(module)) {
      instance <- if (is_nn_module_callable(module)) {
        get_module_instance(module)
      } else if (inherits(module, "nn_Module") || inherits(module, "nn_module")) {
        module
      } else {
        NULL
      }
      if (!is.null(instance)) {
        exec_env[["self"]] <- instance
        fwd <- module$forward
        if (!is.null(fwd)) {
          fwd_formals <- formals(fwd)
          for (nm in names(fwd_formals)) {
            if (!nm %in% names(call_args) && !nm %in% names(exec_env)) {
              default_val <- fwd_formals[[nm]]
              # Skip formals with no default (empty symbol)
              if (is.symbol(default_val) && nchar(as.character(default_val)) == 0) next
              assign(nm, default_val, envir = exec_env)
            }
          }
        }
      }
    }
    parent.env(exec_env) <- asNamespace("Rtorch")

    result <- NULL
    for (stmt in statements) {
      result <- eval(stmt, envir = exec_env)
    }
    result
  }
}


#' Print a Traced Module
#'
#' @param x A traced_module
#' @param ... Ignored
#' @return Invisibly returns x
#' @examples
#' \donttest{
#' m <- nn_linear(3, 2)
#' traced <- trace_module(m, input = torch_randn(c(1, 3)))
#' print(traced)
#' }
#' @export
print.traced_module <- function(x, ...) {
  cat("Traced nn_module\n")
  cat(sprintf("  Parameters: %d\n", length(x$params)))
  cat(sprintf("  Graph breaks: %d\n", length(x$graph_breaks)))

  if (length(x$params) > 0) {
    cat("  Parameter names:\n")
    for (nm in names(x$params)) {
      p <- x$params[[nm]]
      if (inherits(p, "torch_tensor")) {
        cat(sprintf("    %s: %s [%s]\n", nm,
                    paste(p$shape, collapse = "x"),
                    as.character(p$dtype)))
      } else {
        cat(sprintf("    %s\n", nm))
      }
    }
  }

  if (length(x$graph_breaks) > 0) {
    cat("  Graph break reasons:\n")
    for (gb in x$graph_breaks) {
      cat(sprintf("    - %s\n", gb$reason))
    }
  }

  if (length(x$expanded) > 0) {
    cat("  Expanded forward():\n")
    for (stmt in x$expanded) {
      lines <- deparse(stmt, width.cutoff = 80)
      for (ln in lines) cat(sprintf("    %s\n", ln))
    }
  }

  if (!is.null(x$ir)) {
    cat("  IR graph:\n")
    ir_lines <- capture.output(print(x$ir))
    for (ln in ir_lines) cat(sprintf("    %s\n", ln))
  }

  invisible(x)
}
