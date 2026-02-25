#!/usr/bin/env r
#
# codegen.R — Generate Rcpp + R bindings from native_functions.yaml
#
# Usage:
#   r -e 'source("tools/codegen.R"); codegen("path/to/native_functions.yaml")'
#
# Generates:
#   src/gen-ops.cpp   — Rcpp one-liner C++ wrappers
#   R/gen-ops.R       — torch_* namespace functions + method table entries
#
# Strategy:
#   Only generate "simple" ops that fit known templates.
#   Complex ops (custom logic, TensorOptions, etc.) stay hand-written.

# ---- YAML parser (minimal, no dependency) ----

parse_native_functions <- function(path) {
  lines <- readLines(path)
  # Extract func: lines
  func_lines <- grep("^- func:", lines, value = TRUE)
  sigs <- sub("^- func: ", "", func_lines)

  # Also grab variants for each func
  # Build a list of (func_idx, variants) by scanning surrounding lines
  func_idx <- grep("^- func:", lines)
  entries <- vector("list", length(func_idx))

  for (i in seq_along(func_idx)) {
    start <- func_idx[i]
    end <- if (i < length(func_idx)) func_idx[i + 1] - 1L else length(lines)
    block <- lines[start:end]

    sig <- sub("^- func: ", "", block[1])
    variants_line <- grep("^  variants:", block, value = TRUE)
    variants <- if (length(variants_line)) {
      trimws(sub("^  variants:", "", variants_line[1]))
    } else {
      ""
    }
    manual <- any(grepl("manual_cpp_binding: True", block))
    tags_line <- grep("^  tags:", block, value = TRUE)
    tags <- if (length(tags_line)) tags_line[1] else ""

    entries[[i]] <- list(sig = sig, variants = variants, manual = manual,
                         tags = tags)
  }

  entries
}

# ---- Signature parser ----

parse_sig <- function(sig) {
  # Split into name.overload(args) -> return
  m <- regmatches(sig, regexec("^([^(]+)\\((.*)\\)\\s*->\\s*(.+)$", sig))[[1]]
  if (length(m) < 4) return(NULL)

  name_part <- m[2]
  args_str <- m[3]
  ret_str <- trimws(m[4])

  # Split name and overload
  parts <- strsplit(name_part, ".", fixed = TRUE)[[1]]
  name <- parts[1]
  overload <- if (length(parts) > 1) parts[2] else ""

  # Parse args (split on comma, respecting brackets)
  args <- split_args(args_str)

  list(name = name, overload = overload, args = args, ret = ret_str)
}

split_args <- function(s) {
  if (nchar(trimws(s)) == 0) return(list())

  args <- list()
  depth <- 0L
  current <- ""
  keyword_only <- FALSE

  for (ch in strsplit(s, "")[[1]]) {
    if (ch %in% c("(", "[")) {
      depth <- depth + 1L
      current <- paste0(current, ch)
    } else if (ch %in% c(")", "]")) {
      depth <- depth - 1L
      current <- paste0(current, ch)
    } else if (ch == "," && depth == 0L) {
      arg <- parse_one_arg(trimws(current), keyword_only)
      if (!is.null(arg)) {
        if (arg$name == "*") {
          keyword_only <- TRUE
        } else {
          args <- c(args, list(arg))
        }
      }
      current <- ""
    } else {
      current <- paste0(current, ch)
    }
  }
  # Last arg
  current <- trimws(current)
  if (nchar(current) > 0) {
    arg <- parse_one_arg(current, keyword_only)
    if (!is.null(arg) && arg$name != "*") {
      args <- c(args, list(arg))
    }
  }

  args
}

parse_one_arg <- function(s, keyword_only = FALSE) {
  s <- trimws(s)
  if (s == "*") return(list(name = "*", type = "*", default = NULL, kw = FALSE))

  # Strip aliasing annotations: Tensor(a), Tensor(a!), Tensor(a -> *)
  s <- gsub("\\([a-z]+(\\s*->\\s*\\*)?(!)?\\)", "", s)

  # Check for default value
  default <- NULL
  if (grepl("=", s)) {
    eq_pos <- regexpr("=", s)
    default <- trimws(substring(s, eq_pos + 1))
    s <- trimws(substring(s, 1, eq_pos - 1))
  }

  # Split type and name
  parts <- strsplit(s, "\\s+")[[1]]
  if (length(parts) < 2) return(NULL)
  type <- paste(parts[-length(parts)], collapse = " ")
  name <- parts[length(parts)]

  list(name = name, type = type, default = default, kw = keyword_only)
}

# ---- Template classification ----

# Map YAML types to C++ parameter types and R conversion code
TYPE_MAP <- list(
  "Tensor"    = list(cpp = "at::Tensor", r_convert = NULL),
  "Tensor?"   = list(cpp = "SEXP",       r_convert = NULL),  # needs manual
  "Scalar"    = list(cpp = "SEXP",       r_convert = NULL),  # sexp_to_scalar
  "int"       = list(cpp = "int64_t",    r_convert = "as.integer"),
  "float"     = list(cpp = "double",     r_convert = "as.double"),
  "bool"      = list(cpp = "bool",       r_convert = "as.logical"),
  "int?"      = list(cpp = "SEXP",       r_convert = NULL),
  "int[]"     = list(cpp = "SEXP",       r_convert = NULL),
  "int[1]"    = list(cpp = "SEXP",       r_convert = NULL),
  "int[1]?"   = list(cpp = "SEXP",       r_convert = NULL),
  "float?"    = list(cpp = "SEXP",       r_convert = NULL),
  "bool?"     = list(cpp = "SEXP",       r_convert = NULL)
)

# Templates we can auto-generate
# Each template has: pattern match function, C++ generator, R generator

# Simple: all args are Tensor or simple scalars, return is Tensor
classify_op <- function(parsed) {
  if (is.null(parsed)) return("skip")

  # Skip private/internal ops
  if (startsWith(parsed$name, "_")) return("skip")

  # Skip out= variants
  if (grepl("out$", parsed$overload)) return("skip")

  # Skip manual binding ops
  # (checked separately)

  # Return type must be simple
  ret <- parsed$ret
  if (!ret %in% c("Tensor", "bool", "int", "float")) {
    # Allow Tensor with aliasing
    if (!grepl("^Tensor(\\([a-z!]+\\))?$", ret)) return("skip")
  }

  args <- parsed$args

  # Classify by argument pattern
  types <- vapply(args, function(a) a$type, "")

  # Pattern: unary (Tensor self) -> Tensor
  if (length(args) == 1 && types[1] == "Tensor") return("unary")

  # Pattern: binary (Tensor self, Tensor other) -> Tensor
  if (length(args) == 2 && all(types == "Tensor")) return("binary_tt")

  # Pattern: binary (Tensor self, Scalar other) -> Tensor
  if (length(args) == 2 && types[1] == "Tensor" && types[2] == "Scalar") {
    return("binary_ts")
  }

  # Pattern: (Tensor self, Tensor other, Scalar alpha) -> Tensor
  if (length(args) == 3 && types[1] == "Tensor" && types[2] == "Tensor" &&
      types[3] == "Scalar") {
    return("binary_tta")
  }

  # Pattern: reduction with dim (Tensor self, int dim, bool keepdim=False)
  if (length(args) >= 2 && types[1] == "Tensor" && types[2] == "int") {
    remaining <- types[-(1:2)]
    if (all(remaining %in% c("bool", "int"))) return("reduction_dim")
  }

  # Pattern: Tensor + int -> Tensor (e.g., unsqueeze, squeeze with dim)
  if (length(args) == 2 && types[1] == "Tensor" && types[2] == "int") {
    return("tensor_int")
  }

  # Pattern: Tensor + float -> Tensor (e.g., threshold)
  if (length(args) == 2 && types[1] == "Tensor" && types[2] == "float") {
    return("tensor_float")
  }

  # Pattern: Tensor + bool -> Tensor
  if (length(args) == 2 && types[1] == "Tensor" && types[2] == "bool") {
    return("tensor_bool")
  }

  "skip"
}

# ---- C++ code generation ----

gen_cpp_unary <- function(parsed) {
  n <- parsed$name
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(at::Tensor self) { return at::%s(self); }",
    n, n
  )
}

gen_cpp_binary_tt <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  a2 <- args[[2]]$name
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(at::Tensor self, at::Tensor %s) { return at::%s(self, %s); }",
    n, a2, n, a2
  )
}

gen_cpp_binary_ts <- function(parsed) {
  n <- parsed$name
  sprintf(paste0(
    "// [[Rcpp::export]]\n",
    "at::Tensor C_torch_%s(at::Tensor self, SEXP other) {\n",
    "    return at::%s(self, sexp_to_scalar(other));\n",
    "}"
  ), n, n)
}

gen_cpp_binary_tta <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  a2 <- args[[2]]$name
  a3 <- args[[3]]$name
  sprintf(paste0(
    "// [[Rcpp::export]]\n",
    "at::Tensor C_torch_%s(at::Tensor self, at::Tensor %s, SEXP %s_sexp) {\n",
    "    return at::%s(self, %s, sexp_to_scalar(%s_sexp));\n",
    "}"
  ), n, a2, a3, n, a2, a3)
}

gen_cpp_tensor_int <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  a2 <- args[[2]]$name
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(at::Tensor self, int64_t %s) { return at::%s(self, %s); }",
    n, a2, n, a2
  )
}

gen_cpp_tensor_float <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  a2 <- args[[2]]$name
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(at::Tensor self, double %s) { return at::%s(self, %s); }",
    n, a2, n, a2
  )
}

gen_cpp_tensor_bool <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  a2 <- args[[2]]$name
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(at::Tensor self, bool %s) { return at::%s(self, %s); }",
    n, a2, n, a2
  )
}

gen_cpp_reduction_dim <- function(parsed) {
  n <- parsed$name
  args <- parsed$args
  # Build C++ arg list and call list
  cpp_args <- "at::Tensor self"
  call_args <- "self"
  for (i in 2:length(args)) {
    a <- args[[i]]
    if (a$type == "int") {
      cpp_args <- paste0(cpp_args, sprintf(", int64_t %s", a$name))
      call_args <- paste0(call_args, sprintf(", %s", a$name))
    } else if (a$type == "bool") {
      cpp_args <- paste0(cpp_args, sprintf(", bool %s", a$name))
      call_args <- paste0(call_args, sprintf(", %s", a$name))
    }
  }
  sprintf(
    "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) { return at::%s(%s); }",
    n, cpp_args, n, call_args
  )
}

gen_cpp <- function(parsed, template) {
  switch(template,
    unary         = gen_cpp_unary(parsed),
    binary_tt     = gen_cpp_binary_tt(parsed),
    binary_ts     = gen_cpp_binary_ts(parsed),
    binary_tta    = gen_cpp_binary_tta(parsed),
    tensor_int    = gen_cpp_tensor_int(parsed),
    tensor_float  = gen_cpp_tensor_float(parsed),
    tensor_bool   = gen_cpp_tensor_bool(parsed),
    reduction_dim = gen_cpp_reduction_dim(parsed),
    NULL
  )
}

# ---- R code generation ----

gen_r_namespace <- function(parsed, template) {
  n <- parsed$name
  args <- parsed$args

  # Build R function args
  r_formals <- vapply(args, function(a) {
    if (!is.null(a$default)) {
      # Translate Python defaults to R
      d <- a$default
      d <- gsub("True", "TRUE", d)
      d <- gsub("False", "FALSE", d)
      d <- gsub("None", "NULL", d)
      sprintf("%s = %s", a$name, d)
    } else {
      a$name
    }
  }, "")

  # Build C function call args (with conversions)
  c_args <- vapply(args, function(a) {
    if (a$type == "int") {
      sprintf("as.integer(%s)", a$name)
    } else if (a$type == "float") {
      sprintf("as.double(%s)", a$name)
    } else if (a$type == "bool") {
      sprintf("as.logical(%s)", a$name)
    } else if (a$type == "Scalar") {
      a$name  # passed as SEXP, C++ handles conversion
    } else {
      a$name
    }
  }, "")

  # Handle Scalar alpha-style args with _sexp suffix
  c_call_name <- sprintf("C_torch_%s", n)
  c_call <- sprintf("%s(%s)", c_call_name, paste(c_args, collapse = ", "))

  sprintf(
    "#' @export\ntorch_%s <- function(%s) {\n    %s\n}",
    n,
    paste(r_formals, collapse = ", "),
    c_call
  )
}

gen_r_method <- function(parsed, template) {
  n <- parsed$name
  args <- parsed$args

  # Method: first arg is self (implicit), rest are explicit
  if (length(args) <= 1) {
    # Unary: no extra args
    sprintf('.tensor_methods$%s <- function(self) C_torch_%s(self)', n, n)
  } else {
    method_args <- args[-1]
    r_formals <- vapply(method_args, function(a) {
      if (!is.null(a$default)) {
        d <- a$default
        d <- gsub("True", "TRUE", d)
        d <- gsub("False", "FALSE", d)
        d <- gsub("None", "NULL", d)
        sprintf("%s = %s", a$name, d)
      } else {
        a$name
      }
    }, "")

    c_args <- vapply(args, function(a) {
      if (a$type == "int") {
        if (a$name == "self") "self" else sprintf("as.integer(%s)", a$name)
      } else if (a$type == "float") {
        sprintf("as.double(%s)", a$name)
      } else if (a$type == "bool") {
        sprintf("as.logical(%s)", a$name)
      } else {
        a$name
      }
    }, "")

    sprintf(
      '.tensor_methods$%s <- function(self, %s) {\n    C_torch_%s(%s)\n}',
      n,
      paste(r_formals, collapse = ", "),
      n,
      paste(c_args, collapse = ", ")
    )
  }
}

# ---- Exclusion filters ----

# Ops that should never be generated (internal, backend-specific, backward-only)
EXCLUDE_PATTERNS <- c(
  "backward",           # autograd backward ops
  "^fbgemm",            # Facebook GEMM internals
  "^mkldnn",            # Intel MKL-DNN internals
  "^cudnn",             # cuDNN internals
  "^miopen",            # AMD MIOpen internals
  "quantize",           # quantization ops
  "^fake_quantize",     # fake quantization
  "^q_scale$", "^q_zero_point$", "^q_per_channel",  # quantization metadata
  "^int_repr$",         # quantization internal
  "^dequantize$",       # quantization
  "sparse",             # sparse tensor ops
  "^coalesce$", "^is_coalesced$",  # sparse
  "^indices$", "^values$",         # sparse tensor accessors
  "^crow_indices$", "^col_indices$", "^ccol_indices$", "^row_indices$",
  "_copy$",             # copy variants (view_as_real_copy, etc.)
  "^lift", "^lift_fresh",  # functorch internals
  "^is_distributed$",   # distributed
  "^is_same_size$",     # use shape comparison in R
  "^norm_except_dim$",  # internal norm helper
  "^native_norm$",      # internal
  "^nuclear_norm$",     # deprecated
  "^hspmm$",            # sparse matrix multiply
  "^smm$",              # sparse matrix multiply
  "^to_sparse$", "^to_mkldnn",  # backend conversion
  "^dense_dim$", "^sparse_dim$",  # sparse metadata
  "^align_as$",         # named tensor (deprecated feature)
  "^align_to",          # named tensor
  "^numpy_T$",          # Python-specific property
  "^matrix_H$", "^mT$", "^mH$",  # Python-specific properties
  "^resolve_conj$", "^resolve_neg$", "^conj_physical$",  # conjugate internals
  "^one_hot$",          # needs int64 input, not Tensor->Tensor
  "^type_as$",          # use $to() instead
  "^is_set_to$",        # storage internals
  "^infinitely_differentiable_gelu",  # internal
  "^ldexp$",            # deprecated, use mul + pow
  "^matrix_exp_backward$",  # backward
  "^combinations$",     # returns variable-size, tricky
  "^heaviside$",        # niche
  "^expand_as$",        # method-only (self.expand_as), not in at::
  "^reshape_as$",       # method-only
  "^view_as$",          # method-only
  "^unfold$",           # method-only (self.unfold)
  "^equal$",            # returns bool, not Tensor (use eq instead)
  "^is_nonzero$"        # returns bool, not Tensor
)

should_exclude <- function(name) {
  any(vapply(EXCLUDE_PATTERNS, function(pat) grepl(pat, name), FALSE))
}

# ---- Main codegen driver ----

codegen <- function(yaml_path = "~/pytorch-ref/aten/src/ATen/native/native_functions.yaml",
                    out_cpp = "src/gen-ops.cpp",
                    out_r = "R/zzz-gen-ops.R",
                    existing_ops = NULL,
                    dry_run = FALSE) {

  cat("Parsing", yaml_path, "...\n")
  entries <- parse_native_functions(yaml_path)
  cat("Found", length(entries), "function entries\n")

  # If existing_ops not provided, scan current codebase
  if (is.null(existing_ops)) {
    existing_ops <- scan_existing_ops()
  }

  cpp_lines <- character()
  r_ns_lines <- character()
  r_method_lines <- character()
  generated <- character()
  excluded <- character()

  for (entry in entries) {
    if (entry$manual) next

    parsed <- parse_sig(entry$sig)
    if (is.null(parsed)) next

    # Skip in-place variants (trailing _) — generate separately if needed
    if (grepl("_$", parsed$name)) next

    # Apply exclusion filters
    if (should_exclude(parsed$name)) {
      excluded <- c(excluded, parsed$name)
      next
    }

    template <- classify_op(parsed)
    if (template == "skip") next

    # Skip if we already have this op
    if (parsed$name %in% existing_ops) next

    # Skip overloads — only take the primary (empty overload) or first seen
    if (parsed$name %in% generated) next

    # Generate C++
    cpp <- gen_cpp(parsed, template)
    if (is.null(cpp)) next

    cpp_lines <- c(cpp_lines, cpp)

    # Generate R namespace function
    r_ns <- gen_r_namespace(parsed, template)
    r_ns_lines <- c(r_ns_lines, r_ns)

    # Generate R method if variants include "method"
    if (grepl("method", entry$variants)) {
      r_method <- gen_r_method(parsed, template)
      r_method_lines <- c(r_method_lines, r_method)
    }

    generated <- c(generated, parsed$name)
  }

  cat("\nGenerated", length(generated), "new ops\n")
  cat("Already existed:", length(existing_ops), "ops\n")
  cat("Excluded:", length(unique(excluded)), "ops\n")

  if (dry_run) {
    cat("\n=== DRY RUN — would generate ===\n")
    cat("\nC++ ops:\n")
    cat(paste(generated, collapse = ", "), "\n")
    return(invisible(list(
      ops = generated,
      excluded = unique(excluded),
      cpp = cpp_lines,
      r_namespace = r_ns_lines,
      r_methods = r_method_lines
    )))
  }

  # Write C++ file
  cpp_header <- paste0(
    "// Auto-generated by tools/codegen.R — DO NOT EDIT\n",
    "// Source: native_functions.yaml\n",
    "#include \"Rtorch.h\"\n\n"
  )
  writeLines(c(cpp_header, paste(cpp_lines, collapse = "\n\n")), out_cpp)
  cat("Wrote", out_cpp, "\n")

  # Write R file
  r_header <- paste0(
    "# Auto-generated by tools/codegen.R -- DO NOT EDIT\n",
    "# Source: native_functions.yaml\n\n",
    "# ---- Namespace functions ----\n\n"
  )
  r_method_header <- "\n\n# ---- Method table entries ----\n\n"
  writeLines(c(r_header,
               paste(r_ns_lines, collapse = "\n\n"),
               r_method_header,
               paste(r_method_lines, collapse = "\n\n")),
             out_r)
  cat("Wrote", out_r, "\n")

  # Regenerate RcppExports and fix include
  cat("Running Rcpp::compileAttributes...\n")
  Rcpp::compileAttributes(".")
  # compileAttributes writes #include <Rcpp.h> but we need Rtorch.h
  exports_cpp <- readLines("src/RcppExports.cpp")
  exports_cpp <- sub("#include <Rcpp.h>", '#include "Rtorch.h"', exports_cpp,
                     fixed = TRUE)
  writeLines(exports_cpp, "src/RcppExports.cpp")
  cat("Fixed src/RcppExports.cpp include\n")

  invisible(list(ops = generated, cpp = cpp_lines,
                 r_namespace = r_ns_lines, r_methods = r_method_lines))
}

# ---- Scan existing ops to avoid collisions ----

scan_existing_ops <- function(src_dir = "src", r_dir = "R") {
  ops <- character()

  # Scan C++ for C_torch_* function names
  cpp_files <- list.files(src_dir, pattern = "\\.cpp$", full.names = TRUE)
  for (f in cpp_files) {
    if (basename(f) == "gen-ops.cpp") next  # skip our own output
    lines <- readLines(f)
    m <- regmatches(lines, gregexpr("C_torch_(\\w+)", lines))
    ops <- c(ops, unlist(lapply(m, function(x) sub("C_torch_", "", x))))
  }

  # Scan R for C_torch_* calls
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (f in r_files) {
    if (basename(f) == "zzz-gen-ops.R") next
    lines <- readLines(f)
    m <- regmatches(lines, gregexpr("C_torch_(\\w+)", lines))
    ops <- c(ops, unlist(lapply(m, function(x) sub("C_torch_", "", x))))
  }

  unique(ops)
}

# If run directly
if (!interactive() && identical(sys.nframe(), 0L)) {
  result <- codegen(dry_run = TRUE)
}
