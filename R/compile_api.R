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

  # Check fingerprint: if stale, ignore cached kernels
  fingerprint_valid <- FALSE
  if (!is.null(ir) && !is.null(meta$fingerprint)) {
    current_fp <- .artifact_fingerprint(ir)
    fingerprint_valid <- identical(current_fp, meta$fingerprint)
  }

  # Try loading pre-compiled kernels (skip recompilation)
  loaded_kernels <- NULL
  if (fingerprint_valid) {
    loaded_kernels <- tryCatch(.load_kernels(path, ir), error = function(e) NULL)
  }

  # Rebuild prepared graph from IR
  prepared <- NULL
  if (!is.null(ir)) {
    if (!is.null(loaded_kernels) &&
        (length(loaded_kernels$kernels) > 0L ||
         length(loaded_kernels$reduction_kernels) > 0L)) {
      # Fast path: use pre-compiled kernels, skip compilation
      exec_order <- .topo_sort(ir)
      input_map <- list()
      for (inp_id in ir$input_ids) {
        id_str <- as.character(inp_id)
        nm <- ir$nodes[[id_str]]$attrs$name
        if (!is.null(nm)) input_map[[nm]] <- id_str
      }
      prepared <- structure(list(
        graph = ir,
        kernels = loaded_kernels$kernels,
        reduction_kernels = loaded_kernels$reduction_kernels,
        matmul_epilogues = list(),
        fused_node_set = loaded_kernels$fused_node_set,
        exec_order = exec_order,
        input_map = input_map,
        backend = meta$backend %||% "auto"
      ), class = "prepared_graph")
    } else {
      # Slow path: recompile from IR
      dummy_inputs <- list()
      for (nm in names(params)) {
        if (inherits(params[[nm]], "torch_tensor")) {
          dummy_inputs[[nm]] <- params[[nm]]
        }
      }
      for (nm in meta$forward_args) {
        if (!nm %in% names(dummy_inputs)) {
          dummy_inputs[[nm]] <- torch_zeros(1)
        }
      }
      prepared <- tryCatch(
        prepare_graph(ir, dummy_inputs,
                      optimize = FALSE,
                      fuse = TRUE,
                      backend = meta$backend %||% "auto"),
        error = function(e) NULL
      )
    }
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
#' \dontrun{
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

  # Save metadata with fingerprint for cache invalidation
  meta <- compiled$meta
  meta$fingerprint <- .artifact_fingerprint(compiled$ir)
  saveRDS(meta, file.path(path, "meta.rds"))

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

  # Save compiled kernels from prepared graph
  if (!is.null(compiled$prepared)) {
    .save_kernels(compiled$prepared, path)
  }

  invisible(path)
}


#' Build an artifact fingerprint for cache invalidation
#' @noRd
.artifact_fingerprint <- function(ir) {
  rtorch_ver <- as.character(utils::packageVersion("Rtorch"))
  ariel_ver <- tryCatch(
    as.character(utils::packageVersion("ariel")),
    error = function(e) "none"
  )
  sm <- tryCatch(
    as.character(ariel::cuda_sm()),
    error = function(e) "cpu"
  )
  ir_hash <- if (!is.null(ir)) compute_ir_hash(ir) else "none"
  paste(rtorch_ver, ariel_ver, sm, ir_hash, sep = "|")
}


#' Save compiled kernels to artifact directory
#' @noRd
.save_kernels <- function(prepared, path) {
  kernel_dir <- file.path(path, "kernels")
  if (!dir.exists(kernel_dir)) dir.create(kernel_dir)

  kernel_manifest <- list()

  # Save GPU elementwise fusion kernels
  for (gid_str in names(prepared$kernels)) {
    k <- prepared$kernels[[gid_str]]
    if (!isTRUE(k$gpu)) next  # skip CPU kernels (not portable)

    # Extract PTX and metadata from the call_fn closure
    kenv <- environment(k$call_fn)
    if (is.null(kenv)) next

    ptx <- get0("ptx", envir = kenv)
    if (is.null(ptx)) next

    kmeta <- list(
      type = "elementwise",
      group_id = gid_str,
      kernel_name = get0("kernel_name", envir = kenv, ifnotfound = k$func_name),
      shared_mem = get0("shared_mem", envir = kenv, ifnotfound = 0L),
      block_size = get0("block_size", envir = kenv, ifnotfound = 1024L),
      num_warps = get0("num_warps", envir = kenv, ifnotfound = 4L),
      threads_per_block = get0("threads_per_block", envir = kenv, ifnotfound = 128L),
      n_inputs = k$n_inputs %||% 1L,
      external_input_ids = k$external_input_ids,
      output_id = k$output_id,
      group_node_ids = k$group_node_ids
    )

    ptx_file <- file.path(kernel_dir, sprintf("ew_%s.ptx", gid_str))
    meta_file <- file.path(kernel_dir, sprintf("ew_%s.meta.rds", gid_str))
    writeLines(ptx, ptx_file)
    saveRDS(kmeta, meta_file)
    kernel_manifest[[gid_str]] <- list(type = "elementwise", prefix = "ew")
  }

  # Save GPU reduction kernels
  for (id_str in names(prepared$reduction_kernels)) {
    k <- prepared$reduction_kernels[[id_str]]
    kenv <- environment(k$call_fn)
    if (is.null(kenv)) next

    ptx <- get0("ptx", envir = kenv)
    if (is.null(ptx)) next

    kmeta <- list(
      type = "reduction",
      node_id = id_str,
      reduction_type = k$reduction_type,
      kernel_name = get0("kernel_name", envir = kenv, ifnotfound = k$func_name),
      shared_mem = get0("shared_mem", envir = kenv, ifnotfound = 0L),
      block_size = get0("block_size", envir = kenv, ifnotfound = 1024L),
      num_warps = get0("num_warps", envir = kenv, ifnotfound = 4L),
      threads_per_block = get0("threads_per_block", envir = kenv, ifnotfound = 128L),
      external_input_ids = k$external_input_ids,
      output_id = k$output_id
    )

    ptx_file <- file.path(kernel_dir, sprintf("rk_%s.ptx", id_str))
    meta_file <- file.path(kernel_dir, sprintf("rk_%s.meta.rds", id_str))
    writeLines(ptx, ptx_file)
    saveRDS(kmeta, meta_file)
    kernel_manifest[[paste0("rk_", id_str)]] <- list(
      type = "reduction", prefix = "rk"
    )
  }

  # Save manifest
  if (length(kernel_manifest) > 0L) {
    saveRDS(kernel_manifest, file.path(kernel_dir, "manifest.rds"))
  }
}


#' Load pre-compiled kernels from artifact directory
#' @noRd
.load_kernels <- function(path, graph) {
  kernel_dir <- file.path(path, "kernels")
  manifest_file <- file.path(kernel_dir, "manifest.rds")
  if (!file.exists(manifest_file)) return(NULL)

  manifest <- readRDS(manifest_file)
  kernels <- list()
  reduction_kernels <- list()
  fused_node_set <- character()

  for (key in names(manifest)) {
    entry <- manifest[[key]]
    prefix <- entry$prefix

    if (entry$type == "elementwise") {
      gid_str <- sub("^ew_", "", key, perl = TRUE)
      # Use the gid_str directly since manifest stores type = "elementwise"
      # and the file prefix is "ew_<gid_str>"
      ptx_file <- file.path(kernel_dir, sprintf("ew_%s.ptx", gid_str))
      meta_file <- file.path(kernel_dir, sprintf("ew_%s.meta.rds", gid_str))
      if (!file.exists(ptx_file) || !file.exists(meta_file)) next

      ptx <- paste(readLines(ptx_file), collapse = "\n")
      kmeta <- readRDS(meta_file)

      # Rebuild call_fn closure
      call_fn <- .rebuild_elementwise_call_fn(
        ptx = ptx,
        kernel_name = kmeta$kernel_name,
        shared_mem = kmeta$shared_mem,
        block_size = kmeta$block_size,
        threads_per_block = kmeta$threads_per_block
      )

      k <- list(
        call_fn = call_fn,
        func_name = kmeta$kernel_name,
        n_inputs = kmeta$n_inputs,
        gpu = TRUE,
        external_input_ids = kmeta$external_input_ids,
        output_id = kmeta$output_id,
        group_node_ids = kmeta$group_node_ids,
        cache_hit = TRUE
      )
      kernels[[kmeta$group_id]] <- k
      fused_node_set <- c(fused_node_set, as.character(kmeta$group_node_ids))

    } else if (entry$type == "reduction") {
      id_str <- sub("^rk_", "", key, perl = TRUE)
      ptx_file <- file.path(kernel_dir, sprintf("rk_%s.ptx", id_str))
      meta_file <- file.path(kernel_dir, sprintf("rk_%s.meta.rds", id_str))
      if (!file.exists(ptx_file) || !file.exists(meta_file)) next

      ptx <- paste(readLines(ptx_file), collapse = "\n")
      kmeta <- readRDS(meta_file)

      if (kmeta$reduction_type %in% c("softmax", "log_softmax")) {
        call_fn <- .rebuild_softmax_call_fn(
          ptx = ptx,
          kernel_name = kmeta$kernel_name,
          shared_mem = kmeta$shared_mem,
          threads_per_block = kmeta$threads_per_block
        )
      } else if (kmeta$reduction_type == "layer_norm") {
        call_fn <- .rebuild_layer_norm_call_fn(
          ptx = ptx,
          kernel_name = kmeta$kernel_name,
          shared_mem = kmeta$shared_mem,
          threads_per_block = kmeta$threads_per_block
        )
      } else {
        next
      }

      k <- list(
        call_fn = call_fn,
        func_name = kmeta$kernel_name,
        gpu = TRUE,
        reduction_type = kmeta$reduction_type,
        external_input_ids = kmeta$external_input_ids,
        output_id = kmeta$output_id,
        node_id = kmeta$node_id
      )
      reduction_kernels[[id_str]] <- k
      fused_node_set <- c(fused_node_set, id_str)
    }
  }

  list(
    kernels = kernels,
    reduction_kernels = reduction_kernels,
    fused_node_set = fused_node_set
  )
}


#' Rebuild elementwise GPU kernel call_fn from saved PTX
#' @noRd
.rebuild_elementwise_call_fn <- function(ptx, kernel_name, shared_mem,
                                          block_size, threads_per_block) {
  force(ptx); force(kernel_name); force(shared_mem)
  force(block_size); force(threads_per_block)
  function(...) {
    inputs <- list(...)
    ref_idx <- 1L
    ref_numel <- 0L
    for (i in seq_along(inputs)) {
      n <- as.integer(inputs[[i]]$numel())
      if (n > ref_numel) { ref_numel <- n; ref_idx <- i }
    }
    ref_input <- inputs[[ref_idx]]
    n_elem <- ref_numel
    output <- torch_empty_like(ref_input)
    target_shape <- ref_input$shape
    target_ndim <- ref_input$dim()
    for (i in seq_along(inputs)) {
      inp <- inputs[[i]]
      if (as.integer(inp$numel()) != n_elem) {
        while (inp$dim() < target_ndim) inp <- inp$unsqueeze(1L)
        inputs[[i]] <- inp$expand(target_shape)$contiguous()
      }
    }
    grid <- c(as.integer(ceiling(n_elem / block_size)), 1L, 1L)
    block <- c(threads_per_block, 1L, 1L)
    gpu_launch(ptx, kernel_name, inputs, output, grid, block, shared_mem)
    output
  }
}


#' Rebuild softmax GPU kernel call_fn from saved PTX
#' @noRd
.rebuild_softmax_call_fn <- function(ptx, kernel_name, shared_mem,
                                      threads_per_block) {
  force(ptx); force(kernel_name); force(shared_mem); force(threads_per_block)
  function(input) {
    shape <- input$shape
    n_dims <- length(shape)
    n_cols <- shape[n_dims]
    n_rows <- as.integer(prod(shape) / n_cols)
    output <- torch_empty_like(input)
    grid <- c(n_rows, 1L, 1L)
    block <- c(threads_per_block, 1L, 1L)
    gpu_launch_reduction(ptx, kernel_name, input, output,
                          as.integer(n_cols), grid, block, shared_mem)
    output
  }
}


#' Rebuild layer_norm GPU kernel call_fn from saved PTX
#' @noRd
.rebuild_layer_norm_call_fn <- function(ptx, kernel_name, shared_mem,
                                         threads_per_block) {
  force(ptx); force(kernel_name); force(shared_mem); force(threads_per_block)
  function(input, weight, bias) {
    shape <- input$shape
    n_dims <- length(shape)
    n_cols <- shape[n_dims]
    n_rows <- as.integer(prod(shape) / n_cols)
    output <- torch_empty_like(input)
    eps <- 1e-5
    grid <- c(n_rows, 1L, 1L)
    block <- c(threads_per_block, 1L, 1L)
    gpu_launch_generic(ptx, kernel_name,
                        list(input, weight, bias, output),
                        list(as.integer(n_cols), eps),
                        grid, block, shared_mem)
    output
  }
}
