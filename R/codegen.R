#' C++ Code Generation for Fused Kernels
#'
#' Generates vectorized C++ code using ATen's at::vec API,
#' compiles at runtime via Rcpp, and caches compiled kernels.

#' Get libtorch include path, validating ATen headers are present.
#' @return Character path to torch include directory
#' @noRd
.torch_include_path <- function() {
  path <- system.file("include", package = "torch")
  if (!dir.exists(file.path(path, "ATen"))) {
    stop(paste0(
      "libtorch C++ headers not found at: ", path, "/ATen\n",
      "Run torch::install_torch() to download libtorch headers.\n",
      "In non-interactive sessions, set TORCH_INSTALL=1 before loading torch."
    ), call. = FALSE)
  }
  path
}

# Map of torch methods to at::vec C++ expressions
# %s placeholders: input variable name
.vec_op_map <- list(

  # Unary ops
  relu = "at::vec::clamp_min(%s, decltype(%s)(0))",
  sigmoid = "decltype(%s)(1) / (decltype(%s)(1) + (decltype(%s)(0) - %s).exp())",
  tanh = "%s.tanh()",
  exp = "%s.exp()",
  log = "%s.log()",
  sqrt = "%s.sqrt()",
  abs = "%s.abs()",
  neg = "decltype(%s)(0) - %s",


  # Compound activations
  silu = "%s / (decltype(%s)(1) + (decltype(%s)(0) - %s).exp())",

  # Binary ops (need two inputs)
  add = "%s + %s",
  sub = "%s - %s",
  mul = "%s * %s",
  div = "%s / %s"
)

# Scalar versions for tail loop
.scalar_op_map <- list(
  relu = "std::max(%s, decltype(%s)(0))",
  sigmoid = "decltype(%s)(1) / (decltype(%s)(1) + std::exp(-%s))",
  tanh = "std::tanh(%s)",
  exp = "std::exp(%s)",

  log = "std::log(%s)",
  sqrt = "std::sqrt(%s)",
  abs = "std::abs(%s)",
  neg = "-%s",
  silu = "%s / (decltype(%s)(1) + std::exp(-%s))",
  add = "%s + %s",
  sub = "%s - %s",
  mul = "%s * %s",
  div = "%s / %s"
)


#' Generate Vectorized C++ Expression
#'
#' @param op Operation name (e.g., "relu", "add")
#' @param inputs Vector of input variable names
#' @param vectorized Logical, use at::vec or scalar
#' @return C++ expression string
#' @noRd
gen_op_expr <- function(op, inputs, vectorized = TRUE) {
  op_map <- if (vectorized) .vec_op_map else .scalar_op_map

  if (!op %in% names(op_map)) {
    stop(sprintf("Unsupported operation for codegen: %s", op))
  }

  template <- op_map[[op]]

  # Count placeholders
  n_placeholders <- length(gregexpr("%s", template, fixed = TRUE)[[1]])

  # Expand inputs to match placeholders
  if (length(inputs) == 1) {
    args <- rep(inputs, n_placeholders)
  } else {
    args <- inputs
  }

  do.call(sprintf, c(list(template), as.list(args)))
}


#' Extract Operation Chain from Expression
#'
#' Walks a method chain like x$relu()$sigmoid() and extracts ops.
#'
#' @param expr R expression
#' @return List with input_var and ops (in order)
#' @noRd
extract_op_chain <- function(expr) {
  ops <- character()
  current <- expr

  # Walk the chain backwards

  while (is.call(current)) {
    fn <- current[[1]]

    # Method call: something$method()
    if (is.call(fn) && identical(fn[[1]], as.symbol("$"))) {
      method <- as.character(fn[[3]])
      ops <- c(method, ops)  # prepend
      current <- fn[[2]]  # the object
    } else if (is.symbol(fn)) {
      # Regular function call - stop here

      break
    } else {
      break
    }
  }

  # current should now be the input variable

  input_var <- if (is.symbol(current)) as.character(current) else NULL


  list(input_var = input_var, ops = ops)
}


#' Generate Fused Kernel C++ Code
#'
#' @param ops Character vector of operations in order
#' @param input_name Name of input variable
#' @param output_name Name of output variable
#' @param dtype Data type ("float" or "double")
#' @return C++ code string
#' @noRd
gen_fused_kernel_code <- function(ops, input_name = "x", output_name = "out",
                                   dtype = "float", func_name = "fused_kernel") {

  # Validate all ops are supported

  unsupported <- setdiff(ops, names(.vec_op_map))
  if (length(unsupported) > 0) {
    stop(sprintf("Unsupported operations: %s", paste(unsupported, collapse = ", ")))
  }

  vec_type <- sprintf("at::vec::Vectorized<%s>", dtype)

  # Generate vectorized chain
  vec_exprs <- character()
  current_var <- "vx"
  for (i in seq_along(ops)) {
    new_var <- sprintf("v%d", i)
    expr <- gen_op_expr(ops[i], current_var, vectorized = TRUE)
    vec_exprs <- c(vec_exprs, sprintf("    %s %s = %s;", vec_type, new_var, expr))
    current_var <- new_var
  }
  vec_final <- current_var

  # Generate scalar chain for tail
  scalar_exprs <- character()
  current_var <- "sx"
  for (i in seq_along(ops)) {
    new_var <- sprintf("s%d", i)
    expr <- gen_op_expr(ops[i], current_var, vectorized = FALSE)
    scalar_exprs <- c(scalar_exprs, sprintf("      %s %s = %s;", dtype, new_var, expr))
    current_var <- new_var
  }
  scalar_final <- current_var

  # Build kernel using NumericVector interface

  # NOTE: Zero-copy would require RcppTorch to expose XPtrTorchTensor converters.
  # Current approach: tensor → NumericVector → kernel → NumericVector → tensor
  # This adds copy overhead but works reliably.
  code <- sprintf('
#include <Rcpp.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <algorithm>

// [[Rcpp::export]]
Rcpp::NumericVector %s(Rcpp::NumericVector input) {
  int64_t n = input.size();
  Rcpp::NumericVector output(n);

  const double* px = &input[0];
  double* pout = &output[0];

  using Vec = at::vec::Vectorized<double>;
  constexpr int64_t vec_size = Vec::size();
  int64_t vec_end = (n / vec_size) * vec_size;

  // Vectorized loop
  #pragma omp parallel for
  for (int64_t i = 0; i < vec_end; i += vec_size) {
    Vec vx = Vec::loadu(px + i);
%s
    %s.store(pout + i);
  }

  // Scalar tail
  for (int64_t i = vec_end; i < n; i++) {
    double sx = px[i];
%s
    pout[i] = %s;
  }

  return output;
}
',
    func_name,
    paste(gsub(vec_type, "Vec", vec_exprs), collapse = "\n"),
    vec_final,
    paste(gsub(dtype, "double", scalar_exprs), collapse = "\n"),
    scalar_final
  )

  code
}


# Kernel cache environment
.codegen_cache <- new.env(parent = emptyenv())

#' Get Compiled Kernel from Cache
#' @noRd
codegen_cache_get <- function(key) {
  if (exists(key, envir = .codegen_cache, inherits = FALSE)) {
    get(key, envir = .codegen_cache, inherits = FALSE)
  } else {
    NULL
  }
}

#' Store Compiled Kernel in Cache
#' @noRd
codegen_cache_set <- function(key, fn) {
  assign(key, fn, envir = .codegen_cache)
}


#' Compile Fused Kernel
#'
#' Generates and compiles a fused kernel for a chain of operations.
#'
#' @param ops Character vector of operations
#' @param dtype Data type ("float" or "double")
#' @return Compiled function that takes a torch tensor
#' @noRd
compile_fused_kernel <- function(ops, dtype = "float") {
  # Cache key
  key <- paste(c(ops, dtype), collapse = "|")

  # Check cache


  cached <- codegen_cache_get(key)
  if (!is.null(cached)) {
    return(cached)
  }

  # Generate unique function name
  func_name <- sprintf("fused_%s", paste(ops, collapse = "_"))

  # Generate code
  code <- gen_fused_kernel_code(ops, dtype = dtype, func_name = func_name)

  # Get torch paths for compilation
  torch_include <- .torch_include_path()
  torch_lib <- system.file("lib", package = "torch")

  # Set compiler flags
  old_cxxflags <- Sys.getenv("PKG_CXXFLAGS")
  old_libs <- Sys.getenv("PKG_LIBS")

  Sys.setenv(
    PKG_CXXFLAGS = paste0(
      "-I", torch_include, " ",
      "-I", torch_include, "/torch/csrc/api/include ",
      "-D_GLIBCXX_USE_CXX11_ABI=1 ",
      "-O3 -mavx2 -fopenmp "
    ),
    PKG_LIBS = paste0(
      "-L", torch_lib, " ",
      "-ltorch -ltorch_cpu -lc10 -fopenmp ",
      "-Wl,-rpath,", torch_lib
    )
  )

  on.exit({
    Sys.setenv(PKG_CXXFLAGS = old_cxxflags, PKG_LIBS = old_libs)
  })

  # Write and compile
  tmp_file <- tempfile(fileext = ".cpp")
  writeLines(code, tmp_file)

  # Compile via Rcpp
  tryCatch({
    Rcpp::sourceCpp(tmp_file, verbose = FALSE)
    compiled_fn <- get(func_name, envir = globalenv())

    # Cache it
    codegen_cache_set(key, compiled_fn)

    compiled_fn
  }, error = function(e) {
    warning(sprintf("Kernel compilation failed: %s", conditionMessage(e)))
    NULL
  })
}


#' Execute Fused Operation Chain
#'
#' @param tensor Input torch tensor
#' @param ops Character vector of operations
#' @return Output torch tensor
#' @examples
#' \dontrun{
#' fused_ops(torch_randn(c(2, 3)), c("relu", "sigmoid"))
#' }
#' @export
fused_ops <- function(tensor, ops) {
  if (!inherits(tensor, "torch_tensor")) {
    stop("Input must be a torch_tensor")
  }

  # Determine dtype
  dtype_map <- c(
    "Float" = "float",
    "Double" = "double"
  )
  tensor_dtype <- as.character(tensor$dtype)
  dtype <- dtype_map[tensor_dtype]

  if (is.na(dtype)) {
    stop(sprintf("Unsupported dtype for fusion: %s", tensor_dtype))
  }

  # Get or compile kernel
  kernel <- compile_fused_kernel(ops, dtype)

  if (is.null(kernel)) {
    # Fallback to sequential execution
    result <- tensor
    for (op in ops) {
      result <- result[[op]]()
    }
    return(result)
  }

  # Convert tensor to vector, run kernel, convert back
  # NOTE: This legacy path copies data. The IR-based fused kernels
  # in executor.R use zero-copy access via direct libtorch pointers.
  input_shape <- tensor$shape
  input_vec <- as.numeric(tensor$flatten()$cpu())
  output_vec <- kernel(input_vec)
  torch_tensor(output_vec, dtype = tensor$dtype, device = tensor$device)$reshape(input_shape)
}


#' Clear Codegen Cache
#'
#' @return Number of kernels cleared
#' @examples
#' clear_codegen_cache()
#' @export
clear_codegen_cache <- function() {
  n <- length(ls(.codegen_cache))
  rm(list = ls(.codegen_cache), envir = .codegen_cache)
  invisible(n)
}


#' Get Codegen Cache Stats
#'
#' @return List with cache size and keys
#' @examples
#' codegen_cache_stats()
#' @export
codegen_cache_stats <- function() {
  list(
    size = length(ls(.codegen_cache)),
    keys = ls(.codegen_cache)
  )
}


# ============================================================================
# IR-Based Code Generation (Phase 4)
# ============================================================================

# C++ preamble with headers and tensor access helpers
.cpp_preamble <- function() {
  '
#include <torch/torch.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <algorithm>

#include <R.h>
#include <Rinternals.h>

static at::Tensor* get_tensor_ptr(SEXP x) {
  return static_cast<at::Tensor*>(R_ExternalPtrAddr(x));
}

static SEXP make_tensor_sexp(at::Tensor&& t) {
  at::Tensor* p = new at::Tensor(std::move(t));
  SEXP result = PROTECT(R_MakeExternalPtr(p, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(result, [](SEXP x) {
    at::Tensor* ptr = static_cast<at::Tensor*>(R_ExternalPtrAddr(x));
    if (ptr) { delete ptr; R_ClearExternalPtr(x); }
  }, TRUE);
  SEXP cls = PROTECT(Rf_allocVector(STRSXP, 1));
  SET_STRING_ELT(cls, 0, Rf_mkChar("torch_tensor"));
  Rf_setAttrib(result, R_ClassSymbol, cls);
  UNPROTECT(2);
  return result;
}
'
}


#' Get Fusion Group IDs from an Optimized IR Graph
#'
#' @param graph An ir_graph with fusion_group annotations
#' @return Integer vector of unique fusion group IDs (sorted)
#' @noRd
get_fusion_groups <- function(graph) {
  groups <- integer()
  for (node in graph$nodes) {
    gid <- node$attrs$fusion_group
    if (!is.null(gid)) groups <- c(groups, as.integer(gid))
  }
  sort(unique(groups))
}


#' Emit Fused CPU Kernel C++ Code from IR
#'
#' Takes an IR graph and a fusion group ID. Extracts the group's nodes,
#' determines external inputs, and generates a single C++ function with
#' SIMD-vectorized loop + scalar tail using at::vec API.
#'
#' @param graph An ir_graph with fusion annotations
#' @param group_id Integer fusion group ID
#' @param func_name Optional function name (auto-generated if NULL)
#' @return List with code, func_name, n_inputs, external_input_ids,
#'   output_id, group_node_ids, dtype. NULL if group has unsupported ops.
#' @examples
#' \dontrun{
#' stmts <- list(quote(y <- x$relu()$sigmoid()))
#' e <- new.env(); e$x <- torch_randn(c(2, 3))
#' g <- fusion_annotate(lower_to_ir(stmts, e))
#' emit_fused_cpu_kernel(g, 1L)
#' }
#' @export
emit_fused_cpu_kernel <- function(graph, group_id, func_name = NULL) {
  if (!inherits(graph, "ir_graph")) stop("Expected an ir_graph", call. = FALSE)

  # Extract nodes in this fusion group
  group_node_ids <- integer()
  for (id_str in names(graph$nodes)) {
    n <- graph$nodes[[id_str]]
    if (isTRUE(n$attrs$fusion_group == group_id)) {
      group_node_ids <- c(group_node_ids, n$id)
    }
  }
  group_node_ids <- sort(group_node_ids)

  if (length(group_node_ids) == 0L) {
    stop(sprintf("No nodes in fusion group %d", group_id), call. = FALSE)
  }

  group_id_set <- as.character(group_node_ids)

  # Find external inputs (referenced by group nodes but not in the group)
  external_input_ids <- integer()
  for (nid in group_node_ids) {
    node <- graph$nodes[[as.character(nid)]]
    for (inp in node$inputs) {
      if (!as.character(inp) %in% group_id_set) {
        external_input_ids <- c(external_input_ids, inp)
      }
    }
  }
  external_input_ids <- sort(unique(external_input_ids))

  # Output: the last node in the group
  output_id <- max(group_node_ids)

  # Variable naming: external inputs -> "in0", "in1", ...
  ext_names <- list()
  for (i in seq_along(external_input_ids)) {
    ext_names[[as.character(external_input_ids[i])]] <- sprintf("in%d", i - 1L)
  }

  var_name <- function(id) {
    id_str <- as.character(id)
    if (id_str %in% names(ext_names)) return(ext_names[[id_str]])
    sprintf("n%d", id)
  }

  # Check all ops are supported for codegen
  supported_ops <- names(.vec_op_map)
  for (nid in group_node_ids) {
    op <- graph$nodes[[as.character(nid)]]$op
    if (!op %in% supported_ops) {
      return(NULL)  # Can't codegen this group
    }
  }

  # Auto-generate function name
  if (is.null(func_name)) {
    ops <- vapply(group_node_ids, function(nid) {
      graph$nodes[[as.character(nid)]]$op
    }, character(1))
    func_name <- paste0("fused_", paste(ops, collapse = "_"))
    func_name <- gsub("[^a-zA-Z0-9_]", "_", func_name)
  }

  # Determine dtype from first external input's shape annotation
  dtype <- "float"
  if (length(external_input_ids) > 0L) {
    first_ext <- graph$nodes[[as.character(external_input_ids[1])]]
    if (!is.null(first_ext$attrs$output_dtype)) {
      dtype <- switch(first_ext$attrs$output_dtype,
        float32 = "float", float64 = "double", "float")
    }
  }

  # Generate C++ expressions for each node in topological order
  vec_lines <- character()
  scalar_lines <- character()

  for (nid in group_node_ids) {
    node <- graph$nodes[[as.character(nid)]]

    v_inputs <- vapply(node$inputs, function(inp_id) {
      sprintf("v_%s", var_name(inp_id))
    }, character(1))

    s_inputs <- vapply(node$inputs, function(inp_id) {
      sprintf("s_%s", var_name(inp_id))
    }, character(1))

    v_out <- sprintf("v_%s", var_name(nid))
    s_out <- sprintf("s_%s", var_name(nid))

    v_expr <- gen_op_expr(node$op, v_inputs, vectorized = TRUE)
    s_expr <- gen_op_expr(node$op, s_inputs, vectorized = FALSE)

    vec_lines <- c(vec_lines, sprintf("    Vec %s = %s;", v_out, v_expr))
    scalar_lines <- c(scalar_lines,
      sprintf("      %s %s = %s;", dtype, s_out, s_expr))
  }

  out_vec_var <- sprintf("v_%s", var_name(output_id))
  out_scalar_var <- sprintf("s_%s", var_name(output_id))

  # Build function signature
  n_inputs <- length(external_input_ids)
  param_names <- vapply(external_input_ids, var_name, character(1))
  params <- paste(sprintf("SEXP %s_sexp", param_names), collapse = ", ")

  # Input handling lines
  ptr_decls <- character()
  null_checks <- character()
  data_ptrs <- character()
  vec_loads <- character()
  scalar_loads <- character()

  for (i in seq_along(external_input_ids)) {
    vn <- var_name(external_input_ids[i])
    ptr_decls <- c(ptr_decls,
      sprintf("  at::Tensor* %s = get_tensor_ptr(%s_sexp);", vn, vn))
    null_checks <- c(null_checks, sprintf("!%s", vn))
    data_ptrs <- c(data_ptrs,
      sprintf("  %s* p_%s = %s->data_ptr<%s>();", dtype, vn, vn, dtype))
    vec_loads <- c(vec_loads,
      sprintf("    Vec v_%s = Vec::loadu(p_%s + i);", vn, vn))
    scalar_loads <- c(scalar_loads,
      sprintf("      %s s_%s = p_%s[i];", dtype, vn, vn))
  }

  null_check_expr <- paste(null_checks, collapse = " || ")
  first_input <- var_name(external_input_ids[1])

  # Build the full C++ source
  code <- paste0(
    .cpp_preamble(),
    "\nextern \"C\" {\n\n",
    sprintf("SEXP %s(%s) {\n", func_name, params),
    paste(ptr_decls, collapse = "\n"), "\n",
    sprintf(
      "  if (%s) {\n    Rf_error(\"Invalid tensor input\");\n    return R_NilValue;\n  }\n\n",
      null_check_expr),
    sprintf("  at::Tensor output = at::empty_like(*%s);\n", first_input),
    sprintf("  int64_t n = %s->numel();\n", first_input),
    paste(data_ptrs, collapse = "\n"), "\n",
    sprintf("  %s* pout = output.data_ptr<%s>();\n\n", dtype, dtype),
    sprintf("  using Vec = at::vec::Vectorized<%s>;\n", dtype),
    "  constexpr int64_t vec_size = Vec::size();\n",
    "  int64_t vec_end = (n / vec_size) * vec_size;\n\n",
    "  #pragma omp parallel for\n",
    "  for (int64_t i = 0; i < vec_end; i += vec_size) {\n",
    paste(vec_loads, collapse = "\n"), "\n",
    paste(vec_lines, collapse = "\n"), "\n",
    sprintf("    %s.store(pout + i);\n", out_vec_var),
    "  }\n\n",
    "  for (int64_t i = vec_end; i < n; i++) {\n",
    paste(scalar_loads, collapse = "\n"), "\n",
    paste(scalar_lines, collapse = "\n"), "\n",
    sprintf("    pout[i] = %s;\n", out_scalar_var),
    "  }\n\n",
    "  return make_tensor_sexp(std::move(output));\n",
    "}\n\n",
    "} // extern \"C\"\n"
  )

  list(
    code = code,
    func_name = func_name,
    n_inputs = n_inputs,
    external_input_ids = external_input_ids,
    output_id = output_id,
    group_node_ids = group_node_ids,
    dtype = dtype
  )
}


# Kernel disk cache directory
.kernel_cache_dir <- function() {
  d <- file.path(tools::R_user_dir("tinytorch", "cache"), "kernels")
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
  d
}

# In-memory registry of loaded kernel .so files
.kernel_registry <- new.env(parent = emptyenv())

# Hash a string using md5
.hash_code <- function(code) {
  tmp <- tempfile()
  on.exit(unlink(tmp))
  writeLines(code, tmp)
  unname(tools::md5sum(tmp))
}


#' Compile C++ Kernel Source to Shared Library
#'
#' Writes C++ to a temp directory with Makevars, runs R CMD SHLIB,
#' and returns the path to the compiled .so/.dll.
#'
#' @param code C++ source code string
#' @param func_name Function name (used for file naming)
#' @return Path to compiled shared library, or NULL on failure
#' @noRd
.compile_kernel_cpp <- function(code, func_name) {
  build_dir <- tempfile("tinytorch_build_")
  dir.create(build_dir)

  cpp_file <- paste0(func_name, ".cpp")
  writeLines(code, file.path(build_dir, cpp_file))

  # Write Makevars with torch flags
  torch_include <- .torch_include_path()
  torch_lib <- system.file("lib", package = "torch")

  makevars <- sprintf(paste0(
    "PKG_CXXFLAGS = -I%s -I%s/torch/csrc/api/include ",
    "-D_GLIBCXX_USE_CXX11_ABI=1 -O3 -mavx2 -fopenmp\n",
    "PKG_LIBS = -L%s -ltorch -ltorch_cpu -lc10 -fopenmp -Wl,-rpath,%s\n",
    "CXX_STD = CXX17\n"),
    torch_include, torch_include, torch_lib, torch_lib
  )
  writeLines(makevars, file.path(build_dir, "Makevars"))

  # Compile from the build directory
  old_wd <- setwd(build_dir)
  on.exit({
    setwd(old_wd)
  }, add = TRUE)

  out <- tryCatch(
    system2(
      file.path(R.home("bin"), "R"),
      c("CMD", "SHLIB", cpp_file),
      stdout = TRUE, stderr = TRUE
    ),
    error = function(e) {
      attr(e$message, "status") <- 1L
      e$message
    }
  )

  status <- attr(out, "status")
  if (!is.null(status) && status != 0L) {
    warning(sprintf("Kernel compilation failed:\n%s", paste(out, collapse = "\n")),
            call. = FALSE)
    return(NULL)
  }

  so_file <- file.path(build_dir, paste0(func_name, .Platform$dynlib.ext))
  if (!file.exists(so_file)) {
    warning("Compilation produced no shared library", call. = FALSE)
    return(NULL)
  }

  so_file
}


#' Compile a Fusion Group from IR
#'
#' Generates C++, compiles to a shared library with disk caching,
#' and returns a callable kernel.
#'
#' @param graph An ir_graph with fusion annotations
#' @param group_id Integer fusion group ID
#' @return List with call_fn, func_name, n_inputs, external_input_ids,
#'   output_id, group_node_ids, cache_hit. NULL on failure.
#' @noRd
compile_fusion_group <- function(graph, group_id) {
  # Generate C++ code
  kernel_info <- emit_fused_cpu_kernel(graph, group_id)
  if (is.null(kernel_info)) return(NULL)

  code <- kernel_info$code
  func_name <- kernel_info$func_name

  # Check in-memory registry first
  code_hash <- .hash_code(code)
  reg_key <- paste0(code_hash, "|", func_name)

  if (exists(reg_key, envir = .kernel_registry, inherits = FALSE)) {
    info <- get(reg_key, envir = .kernel_registry, inherits = FALSE)
    return(c(info, list(
      external_input_ids = kernel_info$external_input_ids,
      output_id = kernel_info$output_id,
      group_node_ids = kernel_info$group_node_ids,
      cache_hit = TRUE
    )))
  }

  # Check disk cache
  cache_dir <- .kernel_cache_dir()
  cached_so <- file.path(cache_dir, paste0(code_hash, .Platform$dynlib.ext))

  if (file.exists(cached_so)) {
    tryCatch({
      dyn.load(cached_so)
      sym <- getNativeSymbolInfo(func_name)
      call_fn <- function(...) .Call(sym, ...)
      info <- list(call_fn = call_fn, func_name = func_name,
                   n_inputs = kernel_info$n_inputs)
      assign(reg_key, info, envir = .kernel_registry)
      return(c(info, list(
        external_input_ids = kernel_info$external_input_ids,
        output_id = kernel_info$output_id,
        group_node_ids = kernel_info$group_node_ids,
        cache_hit = TRUE
      )))
    }, error = function(e) NULL)
  }

  # Compile from source
  so_path <- .compile_kernel_cpp(code, func_name)
  if (is.null(so_path)) return(NULL)

  # Copy to disk cache
  tryCatch(
    file.copy(so_path, cached_so, overwrite = TRUE),
    error = function(e) NULL
  )

  # Load and register
  tryCatch({
    dyn.load(so_path)
    sym <- getNativeSymbolInfo(func_name)
    call_fn <- function(...) .Call(sym, ...)
    info <- list(call_fn = call_fn, func_name = func_name,
                 n_inputs = kernel_info$n_inputs)
    assign(reg_key, info, envir = .kernel_registry)
    c(info, list(
      external_input_ids = kernel_info$external_input_ids,
      output_id = kernel_info$output_id,
      group_node_ids = kernel_info$group_node_ids,
      cache_hit = FALSE
    ))
  }, error = function(e) {
    warning(sprintf("Failed to load compiled kernel: %s", conditionMessage(e)),
            call. = FALSE)
    NULL
  })
}


#' Clear Kernel Disk Cache
#'
#' Removes all compiled kernel shared libraries from the disk cache
#' and clears the in-memory registry.
#'
#' @return Number of cached kernels cleared (invisibly)
#' @examples
#' clear_kernel_cache()
#' @export
clear_kernel_cache <- function() {
  # Clear in-memory registry
  n <- length(ls(.kernel_registry))
  rm(list = ls(.kernel_registry), envir = .kernel_registry)

  # Clear disk cache
  cache_dir <- .kernel_cache_dir()
  if (dir.exists(cache_dir)) {
    files <- list.files(cache_dir, full.names = TRUE)
    if (length(files) > 0) {
      n <- n + length(files)
      unlink(files)
    }
  }

  invisible(n)
}


#' Get Kernel Cache Statistics
#'
#' @return List with n_memory (in-memory), n_disk (on disk), cache_dir
#' @examples
#' kernel_cache_stats()
#' @export
kernel_cache_stats <- function() {
  cache_dir <- .kernel_cache_dir()
  disk_files <- if (dir.exists(cache_dir)) list.files(cache_dir) else character()
  list(
    n_memory = length(ls(.kernel_registry)),
    n_disk = length(disk_files),
    cache_dir = cache_dir
  )
}


# GPU kernel cache stub (GPU codegen lives in external packages)
.gpu_kernel_cache <- new.env(parent = emptyenv())


# GPU codegen stubs (GPU compilation lives in external packages)
compile_fusion_group_gpu <- function(graph, group_id, ...) NULL
compile_matmul_epilogue_gpu <- function(pattern_info, ...) NULL
compile_softmax_gpu <- function(node_id, node, ...) NULL
compile_layer_norm_gpu <- function(node_id, node, ...) NULL

#' Clear GPU Kernel Cache
#'
#' @return Number of entries cleared (invisibly)
#' @examples
#' clear_gpu_kernel_cache()
#' @export
clear_gpu_kernel_cache <- function() {
  nms <- ls(.gpu_kernel_cache)
  n <- length(nms)
  rm(list = nms, envir = .gpu_kernel_cache)
  invisible(n)
}
