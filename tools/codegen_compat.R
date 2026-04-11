#!/usr/bin/env r
#
# codegen_compat.R — Generate tinytorch wrappers from torch R package source
#
# Reads nn_module, nnf_*, and optimizer definitions from the torch R package
# and generates compatible tinytorch wrappers.
#
# Usage:
#   r -e 'source("tools/codegen_compat.R"); codegen_compat()'
#
# Generates:
#   R/zzz-compat-ops.R — nn_*, nnf_*, optim_* wrappers

# ---- Scan tinytorch for already-implemented ops ----

scan_existing_compat_ops <- function(r_dir = "R") {
  ops <- character()
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (f in r_files) {
    if (basename(f) == "zzz-compat-ops.R") next
    lines <- readLines(f, warn = FALSE)
    # Match: nn_foo <- function, nnf_foo <- function, optim_foo <- function
    m <- regmatches(lines, gregexpr(
      "^(nn_\\w+|nnf_\\w+|optim_\\w+|lr_\\w+)\\s*<-\\s*(function|nn_module|nn_batch_norm)",
      lines))
    for (match in m) {
      if (length(match) > 0) {
        ops <- c(ops, sub("\\s*<-.*", "", match))
      }
    }
    # Also catch: nn_foo <- nn_bar (aliases)
    m2 <- regmatches(lines, gregexpr("^(nn_\\w+|nnf_\\w+)\\s*<-\\s*nn_\\w+$", lines))
    for (match in m2) {
      if (length(match) > 0) ops <- c(ops, sub("\\s*<-.*", "", match))
    }
  }
  unique(ops)
}

# ---- AST extraction from torch source ----

# Extract nn_module definitions from a torch R source file
extract_nn_modules <- function(filepath) {
  exprs <- tryCatch(parse(filepath), error = function(e) {
    message("  Skip (parse error): ", filepath)
    return(expression())
  })

  modules <- list()
  for (i in seq_along(exprs)) {
    e <- exprs[[i]]
    if (!is.call(e) || !identical(e[[1]], as.name("<-"))) next

    lhs <- deparse(e[[2]], width.cutoff = 500)
    rhs <- e[[3]]

    # Match: name <- nn_module("...", ...)
    if (!is.call(rhs)) next
    fn_name <- if (is.call(rhs[[1]])) NULL else as.character(rhs[[1]])
    if (is.null(fn_name) || fn_name != "nn_module") next

    named_args <- as.list(rhs)[-1]
    argnames <- names(named_args)

    # Get classname (first positional arg)
    classname <- if (is.character(named_args[[1]])) named_args[[1]] else lhs

    # Extract initialize and forward
    init_fn <- named_args[["initialize"]]
    fwd_fn <- named_args[["forward"]]
    inherit <- named_args[["inherit"]]

    init_formals <- if (!is.null(init_fn) && is.function(eval(init_fn))) {
      formals(eval(init_fn))
    } else if (!is.null(init_fn)) {
      tryCatch(formals(eval(init_fn)), error = function(e) list())
    } else {
      list()
    }

    fwd_formals <- if (!is.null(fwd_fn) && is.function(eval(fwd_fn))) {
      formals(eval(fwd_fn))
    } else if (!is.null(fwd_fn)) {
      tryCatch(formals(eval(fwd_fn)), error = function(e) list())
    } else {
      list()
    }

    # Deparse forward body
    fwd_body <- if (!is.null(fwd_fn)) {
      tryCatch(deparse(body(eval(fwd_fn)), width.cutoff = 500),
               error = function(e) NULL)
    } else NULL

    # Deparse initialize body
    init_body <- if (!is.null(init_fn)) {
      tryCatch(deparse(body(eval(init_fn)), width.cutoff = 500),
               error = function(e) NULL)
    } else NULL

    modules[[lhs]] <- list(
      name = lhs,
      classname = classname,
      init_formals = init_formals,
      fwd_formals = fwd_formals,
      init_body = init_body,
      fwd_body = fwd_body,
      inherit = if (!is.null(inherit)) deparse(inherit) else NULL
    )
  }
  modules
}

# Extract plain function definitions (nnf_*, etc.)
extract_functions <- function(filepath, prefix = "nnf_") {
  exprs <- tryCatch(parse(filepath), error = function(e) {
    message("  Skip (parse error): ", filepath)
    return(expression())
  })

  fns <- list()
  for (i in seq_along(exprs)) {
    e <- exprs[[i]]
    if (!is.call(e) || !identical(e[[1]], as.name("<-"))) next

    lhs <- deparse(e[[2]], width.cutoff = 500)
    if (!grepl(paste0("^", prefix), lhs)) next

    rhs <- e[[3]]
    if (!is.call(rhs) || !identical(rhs[[1]], as.name("function"))) next

    fn <- tryCatch(eval(rhs), error = function(e) NULL)
    if (is.null(fn)) next

    fn_body <- deparse(body(fn), width.cutoff = 500)
    fn_formals <- formals(fn)

    fns[[lhs]] <- list(
      name = lhs,
      formals = fn_formals,
      body = fn_body
    )
  }
  fns
}

# ---- Code generation ----

# Check if a function body uses internal torch C++ functions
uses_internal_cpp <- function(body_lines) {
  any(grepl("cpp_torch_namespace|cpp_torch_tensor|private\\$|not_implemented_error|torch_tensor_cpp", body_lines))
}

# Format formals as R function signature string
format_formals <- function(fmls) {
  if (length(fmls) == 0) return("")
  # Convert pairlist to regular list to avoid missing-arg errors
  fmls <- as.list(fmls)
  parts <- vapply(names(fmls), function(nm) {
    val <- fmls[[nm]]
    if (missing(val) || (is.name(val) && !nzchar(as.character(val)))) {
      nm
    } else {
      paste0(nm, " = ", deparse(val, width.cutoff = 500))
    }
  }, character(1))
  paste(parts, collapse = ", ")
}

# Generate an nnf_* wrapper
gen_nnf <- function(fn_info) {
  sig <- format_formals(fn_info$formals)
  body <- fn_info$body

  # Check for internal torch calls
  if (uses_internal_cpp(body)) {
    return(c(
      "#' @export",
      sprintf('%s <- function(%s) {', fn_info$name, sig),
      sprintf('  stop("%s is not yet implemented in tinytorch", call. = FALSE)', fn_info$name),
      "}",
      ""
    ))
  }

  # Clean up body: remove torch:: namespace prefixes
  body <- gsub("torch::", "", body)

  c(
    "#' @export",
    sprintf("%s <- function(%s) %s", fn_info$name, sig,
            if (length(body) == 1) body else paste(body, collapse = "\n")),
    ""
  )
}

# Inheritance inlining map: what each base class contributes to initialize
BASE_CLASS_INIT <- list(
  nn_loss = "self$reduction <- reduction",
  nn_weighted_loss = "self$reduction <- reduction; self$weight <- weight",
  nn_dropout_nd = "self$p <- p; self$inplace <- inplace"
)

# Complex base classes that we skip (tinytorch hand-writes these)
COMPLEX_BASES <- c("nn_conv_nd", "nn_rnn_base", "nn_max_pool_nd",
                   "nn_avg_pool_nd", "nn_adaptive_avg_pool_nd",
                   "nn_adaptive_max_pool_nd", "nn_batchnorm")

# Generate an nn_* module constructor
gen_nn_module <- function(mod_info) {
  # Skip if inherits from a complex base
  if (!is.null(mod_info$inherit) && mod_info$inherit %in% COMPLEX_BASES) {
    return(c(
      sprintf("# SKIP: %s inherits from %s (hand-write)", mod_info$name, mod_info$inherit),
      ""
    ))
  }

  # Build initialize body
  init_body <- mod_info$init_body
  if (is.null(init_body) || identical(init_body, "{")) init_body <- "{ }"

  # Check for internal torch calls in init or forward
  all_body <- c(init_body, mod_info$fwd_body %||% character())
  if (uses_internal_cpp(all_body)) {
    sig <- format_formals(mod_info$init_formals)
    return(c(
      "#' @export",
      sprintf('%s <- function(%s) {', mod_info$name, sig),
      sprintf('  stop("%s is not yet implemented in tinytorch", call. = FALSE)', mod_info$name),
      "}",
      ""
    ))
  }

  # Clean bodies
  init_body <- gsub("torch::", "", init_body)
  init_body <- gsub("super\\$initialize\\(([^)]+)\\)", "# parent init: \\1", init_body)

  fwd_body <- mod_info$fwd_body
  if (is.null(fwd_body)) fwd_body <- '  stop("forward not defined")'
  fwd_body <- gsub("torch::", "", fwd_body)

  # Build wrapper function formals (= initialize formals)
  sig <- format_formals(mod_info$init_formals)
  init_sig <- sig  # same

  # Forward formals
  fwd_sig <- format_formals(mod_info$fwd_formals)

  # Arg names for pass-through call
  arg_names <- names(mod_info$init_formals)

  lines <- c(
    "#' @export",
    sprintf("%s <- function(%s) {", mod_info$name, sig),
    sprintf('  nn_module("%s",', mod_info$classname),
    sprintf("    initialize = function(%s) %s,", init_sig,
            if (length(init_body) == 1) init_body
            else paste(init_body, collapse = "\n    ")),
    sprintf("    forward = function(%s) %s", fwd_sig,
            if (length(fwd_body) == 1) fwd_body
            else paste(fwd_body, collapse = "\n    ")),
    sprintf("  )(%s)", paste(arg_names, collapse = ", ")),
    "}",
    ""
  )
  lines
}

# ---- Main driver ----

codegen_compat <- function(torch_src = "~/torch/R",
                           out_r = "R/zzz-compat-ops.R",
                           dry_run = FALSE) {

  cat("Scanning existing tinytorch ops...\n")
  existing <- scan_existing_compat_ops()
  cat("Found", length(existing), "existing nn_*/nnf_*/optim_* ops\n")

  # ---- Extract from torch source ----
  nn_files <- list.files(torch_src, pattern = "^nn-.*\\.R$", full.names = TRUE)
  nnf_files <- list.files(torch_src, pattern = "^nnf-.*\\.R$", full.names = TRUE)

  cat("\nExtracting nn_module definitions from", length(nn_files), "files...\n")
  all_modules <- list()
  for (f in nn_files) {
    mods <- extract_nn_modules(f)
    all_modules <- c(all_modules, mods)
  }
  cat("Found", length(all_modules), "nn_module definitions\n")

  cat("Extracting nnf_* functions from", length(nnf_files), "files...\n")
  all_nnf <- list()
  for (f in nnf_files) {
    fns <- extract_functions(f, prefix = "nnf_")
    all_nnf <- c(all_nnf, fns)
  }
  cat("Found", length(all_nnf), "nnf_* functions\n")

  # ---- Generate ----
  r_lines <- c(
    "# Auto-generated by tools/codegen_compat.R -- DO NOT EDIT",
    sprintf("# Source: torch R package at %s", torch_src),
    sprintf("# Generated: %s", Sys.time()),
    ""
  )

  generated <- character()
  skipped <- character()
  stubbed <- character()
  skip_inherit <- character()

  # nnf_* section
  r_lines <- c(r_lines, "# ---- nnf_* functional wrappers ----", "")
  for (fn in sort(names(all_nnf))) {
    if (fn %in% existing) {
      skipped <- c(skipped, fn)
      next
    }
    code <- gen_nnf(all_nnf[[fn]])
    if (any(grepl("not yet implemented", code))) {
      stubbed <- c(stubbed, fn)
    } else {
      generated <- c(generated, fn)
    }
    r_lines <- c(r_lines, code)
  }

  # nn_* section
  r_lines <- c(r_lines, "# ---- nn_* module constructors ----", "")
  for (mod in sort(names(all_modules))) {
    if (mod %in% existing) {
      skipped <- c(skipped, mod)
      next
    }
    code <- gen_nn_module(all_modules[[mod]])
    if (any(grepl("^# SKIP", code))) {
      skip_inherit <- c(skip_inherit, mod)
    } else if (any(grepl("not yet implemented", code))) {
      stubbed <- c(stubbed, mod)
    } else {
      generated <- c(generated, mod)
    }
    r_lines <- c(r_lines, code)
  }

  # ---- Summary ----
  cat(sprintf("\n=== Summary ===\n"))
  cat(sprintf("Generated: %d\n", length(generated)))
  cat(sprintf("Skipped (already exist): %d\n", length(skipped)))
  cat(sprintf("Stubbed (needs internal torch C++): %d\n", length(stubbed)))
  cat(sprintf("Skipped (complex inheritance): %d\n", length(skip_inherit)))

  if (length(stubbed) > 0) {
    cat("\nStubbed:\n")
    cat(paste("  ", stubbed), sep = "\n")
  }
  if (length(skip_inherit) > 0) {
    cat("\nSkipped (inheritance):\n")
    cat(paste("  ", skip_inherit), sep = "\n")
  }

  if (dry_run) {
    cat("\n=== DRY RUN ===\n")
    return(invisible(list(generated = generated, skipped = skipped,
                          stubbed = stubbed, skip_inherit = skip_inherit)))
  }

  # ---- Write output ----
  writeLines(r_lines, out_r)
  cat(sprintf("\nWrote %d lines to %s\n", length(r_lines), out_r))

  # Validate syntax
  tryCatch({
    parse(out_r)
    cat("Syntax check: OK\n")
  }, error = function(e) {
    warning("Syntax check FAILED: ", conditionMessage(e))
  })

  invisible(list(generated = generated, skipped = skipped,
                 stubbed = stubbed, skip_inherit = skip_inherit))
}
