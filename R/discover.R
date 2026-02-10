#' Module Discovery and Automatic Tracing
#'
#' Walks an nn_module's sub-module tree and reports which modules
#' can be traced by torchlang. Works on any torch-dependent package
#' without the user needing to know the model's internals.

#' Walk an nn_module's Sub-module Tree
#'
#' Recursively discovers all sub-modules in an nn_module instance.
#' Returns a flat list of module info: path, class, parameter count,
#' forward() argument names, and the live module reference.
#'
#' @param module An nn_module instance
#' @param max_depth Maximum recursion depth (default 10)
#' @return A list of module descriptors, each with:
#'   \item{path}{Dot-separated path from root (e.g., "layers.1.mlp")}
#'   \item{class}{Module class name}
#'   \item{n_params}{Number of parameters}
#'   \item{forward_args}{Formal argument names of forward()}
#'   \item{module}{Live reference to the nn_module}
#' @export
discover_modules <- function(module, max_depth = 10L) {
  if (!inherits(module, "nn_module")) {
    stop("module must be an nn_module instance", call. = FALSE)
  }

  result <- list()

  walk <- function(mod, path, depth) {
    if (depth > max_depth) return()

    # Get module class name
    cls <- tryCatch({
      # nn_module stores class name as first element
      cl <- class(mod)
      # R torch modules have class like c("nn_module", "nn_module_generator")
      # The actual name is stored in the module
      name <- mod$.__enclos_env__$private$.module_name
      if (is.null(name) || !nzchar(name)) cl[1] else name
    }, error = function(e) class(mod)[1])

    # Get parameter count
    priv <- tryCatch(mod$.__enclos_env__$private, error = function(e) NULL)
    params <- if (!is.null(priv)) priv$parameters_ else list()
    if (is.null(params)) params <- list()

    # Get forward() formals
    fwd_args <- tryCatch({
      fwd <- mod$forward
      if (is.function(fwd)) {
        nms <- names(formals(fwd))
        # Drop 'self' if present (R6 binds it)
        nms[nms != "self"]
      } else character(0)
    }, error = function(e) character(0))

    # Record this module
    info <- list(
      path = path,
      class = cls,
      n_params = length(params),
      param_names = names(params),
      forward_args = fwd_args,
      module = mod
    )
    result[[length(result) + 1L]] <<- info

    # Recurse into sub-modules
    sub_modules <- if (!is.null(priv)) priv$modules_ else list()
    if (is.null(sub_modules)) sub_modules <- list()

    for (name in names(sub_modules)) {
      sub <- sub_modules[[name]]
      if (inherits(sub, "nn_module")) {
        child_path <- if (nzchar(path)) paste0(path, ".", name) else name
        walk(sub, child_path, depth + 1L)
      }
    }

    # Also check nn_module_list members (stored as numbered elements)
    if (inherits(mod, "nn_module")) {
      tryCatch({
        # nn_module_list stores items in private$modules_ with numeric names
        # Already handled above. But some modules store lists differently.
        NULL
      }, error = function(e) NULL)
    }
  }

  walk(module, "", 0L)
  structure(result, class = "module_tree")
}


#' Print Module Tree
#'
#' @param x A module_tree from discover_modules()
#' @param ... Ignored
#' @export
print.module_tree <- function(x, ...) {
  cat(sprintf("Module tree: %d modules\n\n", length(x)))
  cat(sprintf("%-40s %-25s %6s  %s\n", "Path", "Class", "Params", "Forward args"))
  cat(paste(rep("-", 90), collapse = ""), "\n")

  for (info in x) {
    path <- if (nzchar(info$path)) info$path else "(root)"
    args <- paste(info$forward_args, collapse = ", ")
    cat(sprintf("%-40s %-25s %6d  (%s)\n",
                substr(path, 1, 40), substr(info$class, 1, 25),
                info$n_params, args))
  }
  invisible(x)
}


#' Try Tracing All Discovered Modules
#'
#' Takes a module tree from discover_modules() and attempts trace_module()
#' on each sub-module. Reports graph breaks, correctness, and which modules
#' are candidates for optimization.
#'
#' @param module An nn_module instance (or a module_tree from discover_modules())
#' @param example_inputs Named list of example input tensors for the root module.
#'   For sub-modules, inputs are inferred from forward() signatures using
#'   tensor shapes derived from the root inputs.
#' @param max_depth Maximum depth to trace (default 10)
#' @param check_correctness If TRUE, runs both eager and traced and compares
#'   outputs (default TRUE)
#' @param atol Absolute tolerance for correctness check (default 1e-4)
#' @param verbose Print progress (default TRUE)
#' @return A data.frame with columns: path, class, n_params, graph_breaks,
#'   traceable, correct, error
#' @export
trace_report <- function(module, example_inputs = NULL,
                         max_depth = 10L, check_correctness = TRUE,
                         atol = 1e-4, verbose = TRUE) {
  if (inherits(module, "module_tree")) {
    tree <- module
  } else {
    tree <- discover_modules(module, max_depth = max_depth)
  }

  # Skip root module (path == ""), trace sub-modules only
  # unless root is the only module
  candidates <- if (length(tree) > 1) {
    Filter(function(info) nzchar(info$path), tree)
  } else {
    tree
  }

  # Filter to leaf-ish modules (have forward args and params)
  # Skip container-like modules with no own params
  candidates <- Filter(function(info) {
    length(info$forward_args) > 0
  }, candidates)

  if (verbose) {
    cat(sprintf("Trace report: %d candidate modules\n\n", length(candidates)))
  }

  rows <- list()

  for (info in candidates) {
    path <- if (nzchar(info$path)) info$path else "(root)"
    mod <- info$module

    if (verbose) cat(sprintf("  %-40s ", substr(path, 1, 40)))

    # Try to create example inputs for this sub-module
    inputs <- .make_example_inputs(mod, info$forward_args, example_inputs)

    if (is.null(inputs)) {
      if (verbose) cat("SKIP (can't infer inputs)\n")
      rows[[length(rows) + 1L]] <- data.frame(
        path = path, class = info$class, n_params = info$n_params,
        graph_breaks = NA_integer_, traceable = NA, correct = NA,
        error = "can't infer inputs", stringsAsFactors = FALSE
      )
      next
    }

    # Try tracing
    trace_result <- tryCatch({
      mod$eval()
      do.call(trace_module, c(list(mod), inputs))
    }, error = function(e) {
      conditionMessage(e)
    })

    if (is.character(trace_result)) {
      if (verbose) cat(sprintf("ERROR: %s\n", substr(trace_result, 1, 50)))
      rows[[length(rows) + 1L]] <- data.frame(
        path = path, class = info$class, n_params = info$n_params,
        graph_breaks = NA_integer_, traceable = FALSE, correct = NA,
        error = trace_result, stringsAsFactors = FALSE
      )
      next
    }

    n_breaks <- length(trace_result$graph_breaks)
    traceable <- n_breaks == 0

    # Correctness check
    correct <- NA
    if (check_correctness) {
      correct <- tryCatch({
        ref <- with_no_grad({ do.call(mod, inputs) })
        result <- with_no_grad({ do.call(trace_result$fn, inputs) })

        # Handle list returns
        if (is.list(ref) && !inherits(ref, "torch_tensor")) {
          ref_t <- .first_tensor(ref)
          result_t <- .first_tensor(result)
        } else {
          ref_t <- ref
          result_t <- result
        }

        if (!is.null(ref_t) && !is.null(result_t) &&
            inherits(ref_t, "torch_tensor") && inherits(result_t, "torch_tensor")) {
          as.logical(torch_allclose(ref_t, result_t, atol = atol))
        } else {
          NA
        }
      }, error = function(e) NA)
    }

    status <- if (traceable && isTRUE(correct)) "PASS" else
              if (traceable) sprintf("TRACE-OK (%d breaks)", n_breaks) else
              sprintf("%d breaks", n_breaks)

    if (verbose) {
      ok_str <- if (isTRUE(correct)) "correct" else
                if (is.na(correct)) "?" else "MISMATCH"
      cat(sprintf("%d breaks, %s, %d params\n", n_breaks, ok_str, info$n_params))
    }

    rows[[length(rows) + 1L]] <- data.frame(
      path = path, class = info$class, n_params = info$n_params,
      graph_breaks = n_breaks, traceable = traceable, correct = correct,
      error = NA_character_, stringsAsFactors = FALSE
    )
  }

  result <- do.call(rbind, rows)
  if (is.null(result)) {
    result <- data.frame(
      path = character(0), class = character(0), n_params = integer(0),
      graph_breaks = integer(0), traceable = logical(0), correct = logical(0),
      error = character(0), stringsAsFactors = FALSE
    )
  }

  if (verbose) {
    n_traced <- sum(result$traceable, na.rm = TRUE)
    n_correct <- sum(result$correct, na.rm = TRUE)
    n_total <- nrow(result)
    cat(sprintf("\n  %d/%d traceable, %d/%d correct\n",
                n_traced, n_total, n_correct, n_total))
  }

  result
}


#' Find Installed Packages That Depend on torch
#'
#' Searches installed R packages for those that import or depend on torch.
#'
#' @return Character vector of package names
#' @export
find_torch_packages <- function() {
  ip <- installed.packages()
  pkgs <- character(0)
  for (pkg in c("torch", "Rtorch")) {
    deps <- tools::package_dependencies(pkg, db = ip,
              reverse = TRUE, which = c("Depends", "Imports", "Suggests"))
    found <- deps[[pkg]]
    if (!is.null(found)) pkgs <- c(pkgs, found)
  }
  # Always include Rtorch itself
  pkgs <- unique(c(pkgs, "Rtorch"))
  sort(pkgs)
}


#' Find nn_module Definitions in an R Package
#'
#' Searches a package's R source files for nn_module definitions.
#' Works on both installed packages and local source directories.
#'
#' @param pkg Package name (string) or path to package source directory
#' @return A data.frame with columns: name, file, exported
#' @export
find_modules_in_package <- function(pkg) {
  # Determine source directory
  if (dir.exists(pkg)) {
    r_dir <- file.path(pkg, "R")
    ns_file <- file.path(pkg, "NAMESPACE")
    pkg_name <- basename(pkg)
  } else {
    pkg_path <- find.package(pkg, quiet = TRUE)
    if (length(pkg_path) == 0) {
      stop(sprintf("Package '%s' not found", pkg), call. = FALSE)
    }
    r_dir <- file.path(pkg_path, "R")
    ns_file <- file.path(pkg_path, "NAMESPACE")
    pkg_name <- pkg
  }

  if (!dir.exists(r_dir)) {
    # Installed package — R/ contains .rdb/.rdx, not .R files
    # Try runtime discovery instead
    return(.find_modules_runtime(pkg_name))
  }

  # Find all .R files
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)

  # Search for nn_module definitions
  # Pattern: name <- nn_module(
  pattern <- "^\\s*([a-zA-Z_.][a-zA-Z0-9_.]*)(\\s*<-\\s*)(Rtorch::)?nn_module\\("

  results <- list()
  for (f in r_files) {
    lines <- readLines(f, warn = FALSE)
    matches <- grep(pattern, lines)
    for (m in matches) {
      name <- sub(pattern, "\\1", lines[m])
      name <- trimws(name)
      results[[length(results) + 1L]] <- data.frame(
        name = name,
        file = basename(f),
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(results) == 0) {
    return(data.frame(name = character(0), file = character(0),
                      exported = logical(0), stringsAsFactors = FALSE))
  }

  df <- do.call(rbind, results)

  # Check exports
  exported_names <- character(0)
  if (file.exists(ns_file)) {
    ns_lines <- readLines(ns_file, warn = FALSE)
    export_lines <- grep("^export\\(", ns_lines, value = TRUE)
    exported_names <- sub("^export\\((.*)\\)$", "\\1", export_lines)
  }
  df$exported <- df$name %in% exported_names

  df
}


# ---- Internal helpers ----

#' Make Example Inputs for a Sub-module
#'
#' Tries to create reasonable tensor inputs for a module's forward() args.
#' Uses a small fixed shape (1, 16, dim) where dim is guessed from the
#' module's first parameter's shape.
#'
#' @param mod nn_module instance
#' @param arg_names Character vector of forward() argument names
#' @param root_inputs Optional named list of tensors (from user)
#' @return Named list of tensors, or NULL if we can't figure it out
#' @keywords internal
.make_example_inputs <- function(mod, arg_names, root_inputs = NULL) {
  if (length(arg_names) == 0) return(NULL)

  # Try to find a parameter to guess dimensions
  priv <- tryCatch(mod$.__enclos_env__$private, error = function(e) NULL)
  params <- if (!is.null(priv)) priv$parameters_ else list()
  if (is.null(params)) params <- list()

  # Also check sub-module parameters
  sub_mods <- if (!is.null(priv)) priv$modules_ else list()
  if (is.null(sub_mods)) sub_mods <- list()

  # Collect all params including from sub-modules
  all_params <- params
  for (sm in sub_mods) {
    if (inherits(sm, "nn_module")) {
      sp <- tryCatch(sm$.__enclos_env__$private$parameters_, error = function(e) NULL)
      if (!is.null(sp)) all_params <- c(all_params, sp)
    }
  }

  if (length(all_params) == 0) {
    # Parameterless module (e.g., nn_silu, nn_relu) — use a default shape
    inputs <- list()
    for (nm in arg_names) {
      if (nm %in% c("x", "input", "hidden_states", "sample")) {
        inputs[[nm]] <- torch_randn(1L, 16L, 64L)
      }
    }
    if (length(inputs) > 0) return(inputs)
    return(NULL)
  }

  # Get the first weight parameter to guess input dimension
  first_param <- NULL
  for (p in all_params) {
    if (inherits(p, "torch_tensor") && p$dim() >= 2) {
      first_param <- p
      break
    }
  }
  if (is.null(first_param)) {
    # Try 1D params
    for (p in all_params) {
      if (inherits(p, "torch_tensor") && p$dim() >= 1) {
        first_param <- p
        break
      }
    }
  }
  if (is.null(first_param)) return(NULL)

  # Guess input dimension from weight shape
  # For nn_linear: weight is (out_features, in_features)
  # For nn_conv1d: weight is (out_channels, in_channels, kernel_size)
  # For nn_layer_norm: weight is (normalized_shape,)
  shape <- as.integer(first_param$shape)
  device <- first_param$device

  input_dim <- if (length(shape) >= 2) shape[2] else shape[1]
  batch <- 1L
  seq_len <- 16L

  inputs <- list()
  for (nm in arg_names) {
    # Check if user provided this input
    if (!is.null(root_inputs) && nm %in% names(root_inputs)) {
      inputs[[nm]] <- root_inputs[[nm]]
      next
    }

    # Guess based on common argument names
    if (nm %in% c("x", "input", "hidden_states", "sample")) {
      inputs[[nm]] <- torch_randn(batch, seq_len, input_dim, device = device)
    } else if (nm %in% c("x1", "x2")) {
      inputs[[nm]] <- torch_randn(batch, seq_len, input_dim, device = device)
    } else if (nm %in% c("mask")) {
      inputs[[nm]] <- torch_ones(batch, 1L, seq_len, device = device)
    } else if (nm %in% c("mels")) {
      # Voice encoder style
      inputs[[nm]] <- torch_randn(batch, seq_len * 10L, input_dim, device = device)
    } else if (nm %in% c("attention_mask")) {
      # Optional — skip with NULL
      inputs[[nm]] <- NULL
    } else {
      # Unknown arg — try a tensor of appropriate dim
      inputs[[nm]] <- torch_randn(batch, seq_len, input_dim, device = device)
    }
  }

  # Remove NULL entries (optional args)
  inputs <- Filter(Negate(is.null), inputs)

  if (length(inputs) == 0) return(NULL)
  inputs
}


#' Find First Tensor in a List
#' @keywords internal
.first_tensor <- function(x) {
  if (inherits(x, "torch_tensor")) return(x)
  if (is.list(x)) {
    for (el in x) {
      t <- .first_tensor(el)
      if (!is.null(t)) return(t)
    }
  }
  NULL
}


#' Find nn_modules at Runtime in an Installed Package
#' @keywords internal
.find_modules_runtime <- function(pkg_name) {
  if (!requireNamespace(pkg_name, quietly = TRUE)) {
    stop(sprintf("Package '%s' is not installed", pkg_name), call. = FALSE)
  }

  ns <- asNamespace(pkg_name)
  all_names <- ls(ns, all.names = TRUE)

  # Check exports
  exports <- getNamespaceExports(pkg_name)

  results <- list()
  for (nm in all_names) {
    obj <- tryCatch(get(nm, envir = ns, inherits = FALSE), error = function(e) NULL)
    if (!is.null(obj) && inherits(obj, "nn_module")) {
      results[[length(results) + 1L]] <- data.frame(
        name = nm,
        file = NA_character_,
        exported = nm %in% exports,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(results) == 0) {
    return(data.frame(name = character(0), file = character(0),
                      exported = logical(0), stringsAsFactors = FALSE))
  }

  do.call(rbind, results)
}
