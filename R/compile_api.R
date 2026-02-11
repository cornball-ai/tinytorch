#' Compile and Save/Load Optimized Torch Modules
#'
#' Eagerly compiles nn_modules or plain R functions into optimized
#' computation graphs that can be saved to disk and reloaded without
#' re-tracing.
#'
#' Unlike Python's \code{torch.compile()}, which compiles lazily on
#' first call, R's \code{body()} gives full ASTs without running
#' anything, so we compile eagerly and can save portable artifacts.

#' Compile an nn_module or Function
#'
#' Traces and compiles a torch module or plain R function into an
#' optimized computation graph. Optionally saves the compiled artifact
#' to disk.
#'
#' @param obj An nn_module callable or a plain R function containing
#'   torch operations.
#' @param ... Named example tensors matching the forward() signature
#'   (for modules) or function formals.
#' @param path Optional file path (with \code{.torchlang} extension)
#'   to save the compiled artifact. If non-NULL, the artifact is saved
#'   and the \code{compiled_module} is returned invisibly.
#' @param .optimize Logical, run optimization passes on the IR
#'   (default TRUE).
#' @param .fuse Logical, compile fusion groups to kernels (default TRUE).
#' @param .backend Character: "auto", "gpu", or "cpu". "auto" detects
#'   from input tensors and ariel availability.
#'
#' @return A \code{compiled_module} object (invisibly if \code{path}
#'   is non-NULL).
#'
#' @examples
#' \dontrun{
#' library(torch)
#'
#' # Compile a module
#' model <- nn_linear(10, 5)
#' compiled <- compile(model, input = torch_randn(1, 10))
#' compiled(input = torch_randn(1, 10))
#'
#' # Compile a plain function
#' compiled <- compile(function(x, y) x$matmul(y)$relu(),
#'                     x = torch_randn(3, 3), y = torch_randn(3, 3))
#' compiled(x = torch_randn(3, 3), y = torch_randn(3, 3))
#'
#' # Save/load round-trip
#' compile(model, input = torch_randn(1, 10), path = "/tmp/model.torchlang")
#' loaded <- load("/tmp/model.torchlang")
#' loaded(input = torch_randn(1, 10))
#' }
#'
#' @export
compile <- function(obj, ..., path = NULL,
                    .optimize = TRUE, .fuse = TRUE, .backend = "auto") {
  example_inputs <- list(...)

  if (is_nn_module_callable(obj) ||
      inherits(obj, "nn_Module") || inherits(obj, "nn_module")) {
    compiled <- .compile_module(obj, example_inputs,
                                optimize = .optimize, fuse = .fuse,
                                backend = .backend)
  } else if (is.function(obj)) {
    compiled <- .compile_function(obj, example_inputs,
                                  optimize = .optimize, fuse = .fuse,
                                  backend = .backend)
  } else {
    stop("compile() requires an nn_module or a function", call. = FALSE)
  }

  if (!is.null(path)) {
    .save_artifact(compiled, path)
    return(invisible(compiled))
  }
  compiled
}


#' Load a Compiled Artifact
#'
#' Loads a previously saved \code{.torchlang} artifact from disk,
#' rebuilds kernels from the IR, and returns a callable
#' \code{compiled_module}.
#'
#' @param path Path to a \code{.torchlang} directory.
#'
#' @return A \code{compiled_module} object.
#'
#' @examples
#' \dontrun{
#' loaded <- load_compiled("/tmp/model.torchlang")
#' }
#'
#' @export
load_compiled <- function(path) {
  if (!dir.exists(path)) {
    stop(sprintf("Artifact not found: %s", path), call. = FALSE)
  }

  meta_path <- file.path(path, "meta.rds")
  ir_path <- file.path(path, "ir.rds")
  params_path <- file.path(path, "params.rds")

  if (!file.exists(meta_path)) {
    stop(sprintf("Missing meta.rds in %s", path), call. = FALSE)
  }

  meta <- readRDS(meta_path)
  ir <- if (file.exists(ir_path)) readRDS(ir_path) else NULL
  params <- if (file.exists(params_path)) {
    param_data <- readRDS(params_path)
    lapply(param_data, function(p) {
      torch_tensor(p$data, dtype = structure(p$dtype, class = "torch_dtype"))
    })
  } else {
    list()
  }

  # Rebuild prepared graph from IR
  prepared <- NULL
  if (!is.null(ir)) {
    # Build dummy example inputs from param shapes for prepare_graph.
    # The actual shapes will be re-prepared on first call if different.
    # Use stored forward_args to create zero tensors matching expected names.
    dummy_inputs <- list()
    for (nm in names(params)) {
      if (inherits(params[[nm]], "torch_tensor")) {
        dummy_inputs[[nm]] <- params[[nm]]
      }
    }
    # Add placeholder inputs for forward args (scalar zeros)
    for (nm in meta$forward_args) {
      if (!nm %in% names(dummy_inputs)) {
        dummy_inputs[[nm]] <- torch_zeros(1)
      }
    }
    prepared <- tryCatch(
      prepare_graph(ir, dummy_inputs,
                    optimize = FALSE,  # already optimized
                    fuse = TRUE,
                    backend = meta$backend %||% "auto"),
      error = function(e) NULL
    )
  }

  # Build callable fn using make_traced_fn
  fn <- make_traced_fn(
    statements = meta$expanded,
    params = params,
    input_names = meta$forward_args,
    ir = ir,
    optimize = FALSE,  # already optimized
    fuse = TRUE,
    backend = meta$backend %||% "auto",
    module = NULL
  )

  structure(list(
    fn = fn,
    ir = ir,
    params = params,
    graph_breaks = meta$graph_breaks %||% list(),
    expanded = meta$expanded %||% list(),
    prepared = prepared,
    meta = meta
  ), class = "compiled_module")
}


#' Print a Compiled Module
#'
#' @param x A \code{compiled_module}
#' @param ... Ignored
#' @return Invisibly returns \code{x}
#' @examples
#' \donttest{
#' m <- nn_linear(3, 2)
#' cm <- compile(m, input = torch_randn(c(1, 3)))
#' print(cm)
#' }
#' @export
print.compiled_module <- function(x, ...) {
  cls <- x$meta$class %||% "unknown"
  cat(sprintf("Compiled module: %s\n", cls))
  cat(sprintf("  Parameters: %d\n", length(x$params)))
  cat(sprintf("  Graph breaks: %d\n", length(x$graph_breaks)))
  cat(sprintf("  Backend: %s\n", x$meta$backend %||% "auto"))
  if (!is.null(x$ir)) {
    cat(sprintf("  IR nodes: %d\n", length(x$ir$nodes)))
  }
  if (length(x$graph_breaks) > 0) {
    cat("  Graph break reasons:\n")
    for (gb in x$graph_breaks) {
      cat(sprintf("    - %s\n", gb$reason))
    }
  }
  invisible(x)
}


# ============================================================
# Internal helpers
# ============================================================

#' @noRd
.compile_module <- function(module, example_inputs,
                            optimize, fuse, backend) {
  if (length(example_inputs) == 0) {
    stop("At least one named example input is required", call. = FALSE)
  }
  if (is.null(names(example_inputs)) || any(names(example_inputs) == "")) {
    stop("All example inputs must be named", call. = FALSE)
  }

  # Get class name for metadata
  instance <- if (is_nn_module_callable(module)) {
    get_module_instance(module)
  } else {
    module
  }
  cls_name <- if (is_nn_module_callable(module)) {
    cls <- class(module)
    cls <- cls[!cls %in% c("nn_Module", "nn_module", "R6", "environment")]
    if (length(cls) > 0) cls[1] else "nn_module"
  } else if (!is.null(instance)) {
    cls <- class(instance)
    cls <- cls[!cls %in% c("nn_Module", "nn_module", "R6", "environment")]
    if (length(cls) > 0) cls[1] else "nn_module"
  } else {
    "nn_module"
  }

  # Expand module
  arg_exprs <- lapply(names(example_inputs), as.symbol)
  names(arg_exprs) <- names(example_inputs)
  expanded <- expand_module(module, arg_exprs = arg_exprs)

  # Warn about graph breaks
  if (length(expanded$graph_breaks) > 0) {
    reasons <- vapply(expanded$graph_breaks, function(gb) gb$reason, character(1))
    warning(sprintf("compile: %d graph break(s): %s",
                    length(reasons), paste(reasons, collapse = "; ")),
            call. = FALSE)
  }

  params <- expanded$params

  # Build IR if clean
  ir <- NULL
  if (length(expanded$graph_breaks) == 0 && length(expanded$statements) > 0) {
    ir <- tryCatch({
      g <- lower_to_ir(expanded$statements)
      if (optimize) g <- optimize_graph(g)
      g
    }, error = function(e) NULL)
  }

  # Eagerly prepare the graph for example input shapes
  prepared <- NULL
  if (!is.null(ir)) {
    all_inputs <- c(example_inputs, params)
    ir_input_names <- vapply(ir$input_ids, function(id) {
      ir$nodes[[as.character(id)]]$attrs$name %||% ""
    }, character(1))
    inputs_for_ir <- all_inputs[ir_input_names]

    resolved_backend <- if (backend == "auto") {
      .detect_backend(example_inputs)
    } else {
      backend
    }

    prepared <- tryCatch(
      prepare_graph(ir, inputs_for_ir,
                    optimize = FALSE,  # already optimized above
                    fuse = fuse, backend = resolved_backend),
      error = function(e) NULL
    )
  }

  # Build callable fn (reuse make_traced_fn from trace.R)
  fn <- make_traced_fn(
    statements = expanded$statements,
    params = params,
    input_names = names(example_inputs),
    ir = ir,
    optimize = optimize,
    fuse = fuse,
    backend = backend,
    module = module
  )

  meta <- list(
    class = cls_name,
    forward_args = names(example_inputs),
    graph_breaks = expanded$graph_breaks,
    expanded = expanded$statements,
    backend = backend,
    pkg_version = as.character(utils::packageVersion("Rtorch"))
  )

  structure(list(
    fn = fn,
    ir = ir,
    params = params,
    graph_breaks = expanded$graph_breaks,
    expanded = expanded$statements,
    prepared = prepared,
    meta = meta
  ), class = "compiled_module")
}


#' @noRd
.compile_function <- function(fn, example_inputs,
                              optimize, fuse, backend) {
  if (length(example_inputs) == 0) {
    stop("At least one named example input is required", call. = FALSE)
  }
  if (is.null(names(example_inputs)) || any(names(example_inputs) == "")) {
    stop("All example inputs must be named", call. = FALSE)
  }

  fn_body <- body(fn)
  if (is.null(fn_body)) {
    stop("Function has no body", call. = FALSE)
  }

  # Extract statements from body
  statements <- if (is.call(fn_body) && identical(fn_body[[1]], as.symbol("{"))) {
    as.list(fn_body[-1])
  } else {
    list(fn_body)
  }

  # Lower to IR
  ir <- tryCatch({
    g <- lower_to_ir(statements)
    if (optimize) g <- optimize_graph(g)
    g
  }, error = function(e) NULL)

  params <- list()  # plain functions have no parameters

  # Eagerly prepare
  prepared <- NULL
  if (!is.null(ir)) {
    ir_input_names <- vapply(ir$input_ids, function(id) {
      ir$nodes[[as.character(id)]]$attrs$name %||% ""
    }, character(1))
    inputs_for_ir <- example_inputs[ir_input_names]

    resolved_backend <- if (backend == "auto") {
      .detect_backend(example_inputs)
    } else {
      backend
    }

    prepared <- tryCatch(
      prepare_graph(ir, inputs_for_ir,
                    optimize = FALSE,
                    fuse = fuse, backend = resolved_backend),
      error = function(e) NULL
    )
  }

  # Build callable fn
  fn_closure <- make_traced_fn(
    statements = statements,
    params = params,
    input_names = names(example_inputs),
    ir = ir,
    optimize = optimize,
    fuse = fuse,
    backend = backend,
    module = NULL
  )

  meta <- list(
    class = "function",
    forward_args = names(example_inputs),
    graph_breaks = list(),
    expanded = statements,
    backend = backend,
    pkg_version = as.character(utils::packageVersion("Rtorch"))
  )

  structure(list(
    fn = fn_closure,
    ir = ir,
    params = params,
    graph_breaks = list(),
    expanded = statements,
    prepared = prepared,
    meta = meta
  ), class = "compiled_module")
}


#' @noRd
.save_artifact <- function(compiled, path) {
  if (!grepl("\\.torchlang$", path)) {
    stop("path must have .torchlang extension", call. = FALSE)
  }
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE)
  }

  # Save metadata
  saveRDS(compiled$meta, file.path(path, "meta.rds"))

  # Save IR (if available)
  if (!is.null(compiled$ir)) {
    saveRDS(compiled$ir, file.path(path, "ir.rds"))
  }

  # Save parameters as R arrays (pure-R serialization, no torch dependency)
  if (length(compiled$params) > 0) {
    tensor_params <- Filter(function(p) inherits(p, "torch_tensor"), compiled$params)
    if (length(tensor_params) > 0) {
      param_data <- lapply(tensor_params, function(t) {
        list(data = as.array(t),
             dtype = unclass(t$dtype),
             shape = as.integer(t$shape))
      })
      saveRDS(param_data, file.path(path, "params.rds"))
    }
  }

  invisible(path)
}
