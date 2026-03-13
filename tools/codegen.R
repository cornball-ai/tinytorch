#!/usr/bin/env r
#
# codegen.R — Generate Rcpp + R bindings from native_functions.yaml
#
# Usage:
#   r -e 'source("tools/codegen.R"); codegen()'
#
# Generates:
#   src/gen-ops.cpp   — Rcpp C++ wrappers
#   R/zzz-gen-ops.R   — torch_* namespace functions + method table entries

# ---- YAML parser (minimal, no dependency) ----

parse_native_functions <- function(path) {
  lines <- readLines(path)
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

    is_out <- is_out_variant(sig)

    entries[[i]] <- list(sig = sig, variants = variants, manual = manual,
                         tags = tags, is_out = is_out)
  }

  entries
}

# Detect out-variant signatures by looking for output tensor args.
# Out variants have Tensor(a!) args after the keyword-only marker (*).
is_out_variant <- function(sig) {
  m <- regmatches(sig, regexec("\\((.*)\\)", sig))[[1]]
  if (length(m) < 2) return(FALSE)
  args_str <- m[2]
  if (grepl("\\*.*Tensor\\([a-z]+!\\)", args_str)) return(TRUE)
  FALSE
}

# ---- Signature parser ----

parse_sig <- function(sig) {
  m <- regmatches(sig, regexec("^([^(]+)\\((.*)\\)\\s*->\\s*(.+)$", sig))[[1]]
  if (length(m) < 4) return(NULL)

  name_part <- m[2]
  args_str <- m[3]
  ret_str <- trimws(m[4])

  parts <- strsplit(name_part, ".", fixed = TRUE)[[1]]
  name <- parts[1]
  overload <- if (length(parts) > 1) parts[2] else ""

  args <- split_args(args_str)

  # Normalize SymInt -> int, SymInt[] -> int[]
  for (i in seq_along(args)) {
    args[[i]]$type <- gsub("SymInt", "int", args[[i]]$type)
  }

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

  default <- NULL
  if (grepl("=", s)) {
    eq_pos <- regexpr("=", s)
    default <- trimws(substring(s, eq_pos + 1))
    s <- trimws(substring(s, 1, eq_pos - 1))
  }

  parts <- strsplit(s, "\\s+")[[1]]
  if (length(parts) < 2) return(NULL)
  type <- paste(parts[-length(parts)], collapse = " ")
  name <- parts[length(parts)]

  list(name = name, type = type, default = default, kw = keyword_only)
}

# ---- Per-argument type conversion table ----
#
# Each entry: list(cpp_type, cpp_call_expr, r_convert)
#   cpp_type: C++ parameter type in function signature
#   cpp_call_expr: expression to pass to at::func(); NULL means just use param name
#   r_convert: R-side conversion in wrapper; NULL means pass through

ARG_CONVERSIONS <- list(
  "Tensor" = list(
    cpp_type = "at::Tensor",
    cpp_call = NULL,
    r_convert = NULL
  ),
  "Scalar" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_scalar(%s)",
    r_convert = NULL
  ),
  "int" = list(
    cpp_type = "int64_t",
    cpp_call = NULL,
    r_convert = "as.integer(%s)"
  ),
  "float" = list(
    cpp_type = "double",
    cpp_call = NULL,
    r_convert = "as.double(%s)"
  ),
  "bool" = list(
    cpp_type = "bool",
    cpp_call = NULL,
    r_convert = "as.logical(%s)"
  ),
  "str" = list(
    cpp_type = "std::string",
    cpp_call = NULL,
    r_convert = NULL
  ),
  "ScalarType?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_dtype(%s)",
    r_convert = NULL
  ),
  "Tensor?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_tensor(%s)",
    r_convert = NULL
  ),
  "int?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_int(%s)",
    r_convert = NULL
  ),
  "float?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_double(%s)",
    r_convert = NULL
  ),
  "bool?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_bool(%s)",
    r_convert = NULL
  ),
  "str?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_string(%s)",
    r_convert = NULL
  ),
  "Scalar?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_scalar(%s)",
    r_convert = NULL
  ),
  "Device?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_device(%s)",
    r_convert = NULL
  ),
  "Device" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_required_device(%s)",
    r_convert = NULL
  ),
  "ScalarType" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_dtype(%s).value()",
    r_convert = NULL
  ),
  "Generator?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_generator(%s)",
    r_convert = NULL
  ),
  "MemoryFormat?" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_memory_format(%s)",
    r_convert = NULL
  ),
  "MemoryFormat" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_optional_memory_format(%s).value()",
    r_convert = NULL
  ),
  "Layout?" = list(
    cpp_type = "SEXP",
    cpp_call = NULL,
    r_convert = NULL,
    skip_in_call = TRUE
  ),
  "Dimname" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_dimname(%s)",
    r_convert = NULL
  )
)

# int[] patterns (int[], int[1], int[2], int[3], etc.) and optional variants
is_int_array_type <- function(type) {
  grepl("^int\\[\\d*\\]\\??$", type)
}

# float[] patterns (float[], float[]?)
is_float_array_type <- function(type) {
  grepl("^float\\[\\]\\??$", type)
}

# Tensor[] pattern
is_tensor_list_type <- function(type) {
  type == "Tensor[]"
}

# Scalar[] pattern
is_scalar_list_type <- function(type) {
  type == "Scalar[]"
}

# Tensor?[] pattern
is_optional_tensor_list_type <- function(type) {
  type == "Tensor?[]"
}

# Dimname[] patterns (Dimname[], Dimname[1], Dimname[]?)
is_dimname_array_type <- function(type) {
  grepl("^Dimname\\[\\d*\\]\\??$", type)
}

# ---- Return type support ----

# Parse return type string. Returns list(types, names) for tuples,
# or list(types = "Tensor", names = NULL) for simple.
parse_return_type <- function(ret_str) {
  ret_str <- trimws(ret_str)

  # Strip aliasing from simple return: Tensor(a) -> Tensor
  if (grepl("^Tensor(\\([a-z!]+\\))?$", ret_str)) {
    return(list(types = "Tensor", names = NULL))
  }

  # Void return: ()
  if (ret_str == "()") {
    return(list(types = "void", names = NULL))
  }

  # Tuple: (Tensor foo, Tensor bar) or mixed tuples
  if (grepl("^\\(", ret_str)) {
    inner <- sub("^\\((.*)\\)$", "\\1", ret_str)
    parts <- strsplit(inner, ",")[[1]]
    types <- character(length(parts))
    names <- character(length(parts))
    for (i in seq_along(parts)) {
      p <- trimws(parts[i])
      # Strip aliasing
      p <- gsub("\\([a-z!]+\\)", "", p)
      tokens <- strsplit(trimws(p), "\\s+")[[1]]
      types[i] <- tokens[1]
      names[i] <- if (length(tokens) > 1) tokens[2] else paste0("r", i)
    }
    return(list(types = types, names = names))
  }

  # Simple non-Tensor types
  if (ret_str == "bool") return(list(types = "bool", names = NULL))
  if (ret_str == "int") return(list(types = "int", names = NULL))
  if (ret_str == "float") return(list(types = "float", names = NULL))
  if (ret_str == "Scalar") return(list(types = "Scalar", names = NULL))
  if (ret_str == "ScalarType") return(list(types = "ScalarType", names = NULL))
  if (ret_str == "QScheme") return(list(types = "QScheme", names = NULL))
  if (ret_str == "Tensor[]") return(list(types = "Tensor[]", names = NULL))

  # Tensor(a)[] — list of aliased tensors
  if (grepl("^Tensor(\\([a-z!]+\\))?\\[\\]$", ret_str)) {
    return(list(types = "Tensor[]", names = NULL))
  }

  NULL
}

# ---- Generability check ----

# Is this argument type supported by codegen?
is_supported_arg <- function(type) {
  if (type %in% names(ARG_CONVERSIONS)) return(TRUE)
  if (is_int_array_type(type)) return(TRUE)
  if (is_float_array_type(type)) return(TRUE)
  if (is_tensor_list_type(type)) return(TRUE)
  if (is_scalar_list_type(type)) return(TRUE)
  if (is_optional_tensor_list_type(type)) return(TRUE)
  if (is_dimname_array_type(type)) return(TRUE)
  FALSE
}

# Is this return type supported?
is_supported_return <- function(ret) {
  if (is.null(ret)) return(FALSE)
  # Simple Tensor return
  if (length(ret$types) == 1 && ret$types == "Tensor") return(TRUE)
  # Tensor list return
  if (length(ret$types) == 1 && ret$types == "Tensor[]") return(TRUE)
  # Void return
  if (length(ret$types) == 1 && ret$types == "void") return(TRUE)
  # Scalar types
  if (length(ret$types) == 1 && ret$types %in% c("bool", "int", "float",
                                                   "Scalar", "ScalarType",
                                                   "QScheme"))
    return(TRUE)
  # Tuple of Tensors
  if (length(ret$types) > 1 && all(ret$types == "Tensor")) return(TRUE)
  # Mixed tuple with Tensor and Tensor[]
  if (length(ret$types) > 1 && all(ret$types %in% c("Tensor", "Tensor[]")))
    return(TRUE)
  FALSE
}

# TensorOptions quartet detection
TENSOR_OPTIONS_NAMES <- c("dtype", "layout", "device", "pin_memory")

has_tensor_options <- function(args) {
  arg_names <- vapply(args, function(a) a$name, "")
  all(c("dtype", "device") %in% arg_names) &&
    any(vapply(args, function(a) a$type == "ScalarType?", FALSE))
}

# Types we still cannot generate
SKIP_ARG_TYPES <- c("Storage", "Stream")

can_generate <- function(parsed) {
  if (is.null(parsed)) return(FALSE)

  ret <- parse_return_type(parsed$ret)
  if (!is_supported_return(ret)) return(FALSE)

  for (a in parsed$args) {
    # Skip the TensorOptions args we'll handle specially
    if (a$name %in% c("layout", "pin_memory") &&
        a$type %in% c("Layout?", "bool?")) next
    if (a$type %in% SKIP_ARG_TYPES) return(FALSE)
    if (!is_supported_arg(a$type)) return(FALSE)
  }

  TRUE
}

# ---- C++ code generation (general emitter) ----

# Get the C++ parameter declaration for an arg
cpp_param <- function(a) {
  type <- a$type
  name <- a$name

  # Skip TensorOptions layout/pin_memory
  if (name %in% c("layout", "pin_memory")) return(NULL)

  if (type %in% names(ARG_CONVERSIONS)) {
    conv <- ARG_CONVERSIONS[[type]]
    # Scalar args get _sexp suffix to avoid name collision with call expr
    if (type == "Scalar") {
      return(sprintf("%s %s_sexp", conv$cpp_type, name))
    }
    return(sprintf("%s %s", conv$cpp_type, name))
  }

  if (is_int_array_type(type)) return(sprintf("SEXP %s_sexp", name))
  if (is_float_array_type(type)) return(sprintf("SEXP %s_sexp", name))
  if (is_tensor_list_type(type)) return(sprintf("SEXP %s_sexp", name))
  if (is_scalar_list_type(type)) return(sprintf("SEXP %s_sexp", name))
  if (is_optional_tensor_list_type(type)) return(sprintf("SEXP %s_sexp", name))
  if (is_dimname_array_type(type)) return(sprintf("SEXP %s_sexp", name))

  NULL
}

# Get the C++ call expression for an arg
cpp_call_expr <- function(a) {
  type <- a$type
  name <- a$name

  if (name %in% c("layout", "pin_memory")) return(NULL)

  if (type %in% names(ARG_CONVERSIONS)) {
    conv <- ARG_CONVERSIONS[[type]]
    if (isTRUE(conv$skip_in_call)) return(NULL)
    if (!is.null(conv$cpp_call)) {
      actual_name <- if (type == "Scalar") paste0(name, "_sexp") else name
      return(sprintf(conv$cpp_call, actual_name))
    }
    return(name)
  }

  if (is_int_array_type(type)) {
    is_optional <- grepl("\\?$", type)
    if (is_optional) return(sprintf("%s_ref", name))
    return(sprintf("at::IntArrayRef(%s_vec.data(), %s_vec.size())", name, name))
  }

  if (is_float_array_type(type)) {
    is_optional <- grepl("\\?$", type)
    if (is_optional) return(sprintf("%s_ref", name))
    return(sprintf("at::ArrayRef<double>(%s_vec.data(), %s_vec.size())", name, name))
  }

  if (is_tensor_list_type(type)) return(sprintf("%s_vec", name))
  if (is_scalar_list_type(type)) return(sprintf("%s_vec", name))
  if (is_optional_tensor_list_type(type)) return(sprintf("%s_vec", name))

  if (is_dimname_array_type(type)) {
    is_optional <- grepl("\\?$", type)
    if (is_optional) return(sprintf("%s_ref", name))
    return(sprintf("at::DimnameList(%s_vec.data(), %s_vec.size())", name, name))
  }

  name
}

# Generate locals needed before the call (int[] vectors, Tensor[] lists, etc.)
cpp_locals <- function(args) {
  lines <- character()
  for (a in args) {
    if (is_int_array_type(a$type)) {
      is_optional <- grepl("\\?$", a$type)
      if (is_optional) {
        lines <- c(lines,
          sprintf("    c10::optional<at::IntArrayRef> %s_ref;", a$name),
          sprintf("    std::vector<int64_t> %s_vec;", a$name),
          sprintf("    if (!Rf_isNull(%s_sexp)) {", a$name),
          sprintf("        %s_vec = sexp_to_int_vec(%s_sexp);", a$name, a$name),
          sprintf("        %s_ref = at::IntArrayRef(%s_vec.data(), %s_vec.size());",
                  a$name, a$name, a$name),
          "    }"
        )
      } else {
        lines <- c(lines, sprintf("    auto %s_vec = sexp_to_int_vec(%s_sexp);",
                                  a$name, a$name))
      }
    }
    if (is_float_array_type(a$type)) {
      is_optional <- grepl("\\?$", a$type)
      if (is_optional) {
        lines <- c(lines,
          sprintf("    c10::optional<at::ArrayRef<double>> %s_ref;", a$name),
          sprintf("    std::vector<double> %s_vec;", a$name),
          sprintf("    if (!Rf_isNull(%s_sexp)) {", a$name),
          sprintf("        %s_vec = sexp_to_double_vec(%s_sexp);", a$name, a$name),
          sprintf("        %s_ref = at::ArrayRef<double>(%s_vec.data(), %s_vec.size());",
                  a$name, a$name, a$name),
          "    }"
        )
      } else {
        lines <- c(lines, sprintf("    auto %s_vec = sexp_to_double_vec(%s_sexp);",
                                  a$name, a$name))
      }
    }
    if (is_tensor_list_type(a$type)) {
      lines <- c(lines, sprintf("    auto %s_vec = sexp_to_tensor_list(%s_sexp);",
                                a$name, a$name))
    }
    if (is_scalar_list_type(a$type)) {
      lines <- c(lines, sprintf("    auto %s_vec = sexp_to_scalar_list(%s_sexp);",
                                a$name, a$name))
    }
    if (is_optional_tensor_list_type(a$type)) {
      lines <- c(lines,
        sprintf("    auto %s_vec = sexp_to_optional_tensor_list(%s_sexp);",
                a$name, a$name))
    }
    if (is_dimname_array_type(a$type)) {
      is_optional <- grepl("\\?$", a$type)
      if (is_optional) {
        lines <- c(lines,
          sprintf("    c10::optional<at::DimnameList> %s_ref;", a$name),
          sprintf("    std::vector<at::Dimname> %s_vec;", a$name),
          sprintf("    if (!Rf_isNull(%s_sexp)) {", a$name),
          sprintf("        %s_vec = sexp_to_dimname_vec(%s_sexp);", a$name, a$name),
          sprintf("        %s_ref = at::DimnameList(%s_vec.data(), %s_vec.size());",
                  a$name, a$name, a$name),
          "    }"
        )
      } else {
        lines <- c(lines, sprintf("    auto %s_vec = sexp_to_dimname_vec(%s_sexp);",
                                  a$name, a$name))
      }
    }
  }
  lines
}

# Ops that exist only as Tensor methods, not as at:: free functions.
# Their at::ops/*.h headers have an empty at:: namespace block.
METHOD_ONLY_OPS <- c(
  "chalf", "coalesce", "contiguous", "data", "expand_as", "indices",
  "is_coalesced", "is_set_to", "item", "mH", "mT", "numpy_T",
  "matrix_H", "qscheme", "rename", "rename_", "reshape_as", "to",
  "to_dense", "to_sparse", "to_padded_tensor", "type_as",
  "unfold", "values", "view_as",
  "sum_to_size", "pin_memory",
  "new_empty", "new_empty_strided", "new_full", "new_zeros", "new_ones",
  "dense_dim", "sparse_dim",
  "is_same_size", "is_distributed", "is_nonzero"
)

# Build the call expression for an op, either at::name() or self.name()
build_call <- function(name, active_args, variants, skip_self = FALSE) {
  is_method_only <- grepl("method", variants) && !grepl("function", variants)
  if (name %in% METHOD_ONLY_OPS) is_method_only <- TRUE

  if (is_method_only || skip_self) {
    # Method call: self.op(other_args)
    self_idx <- which(vapply(active_args, function(a) a$name == "self", FALSE))
    if (length(self_idx) == 0) self_idx <- 1L
    non_self_args <- if (length(active_args) > 1) active_args[-self_idx[1]] else list()
    call_parts <- vapply(non_self_args, function(a) {
      e <- cpp_call_expr(a)
      if (is.null(e)) "" else e
    }, "")
    call_parts <- call_parts[nzchar(call_parts)]
    call_str <- paste(call_parts, collapse = ", ")
    locals <- cpp_locals(non_self_args)
    list(expr = sprintf("self.%s(%s)", name, call_str),
         locals = locals, style = "method")
  } else {
    # Free function: at::name(all_args)
    call_parts <- vapply(active_args, function(a) {
      e <- cpp_call_expr(a)
      if (is.null(e)) "" else e
    }, "")
    call_parts <- call_parts[nzchar(call_parts)]
    call_str <- paste(call_parts, collapse = ", ")
    locals <- cpp_locals(active_args)
    list(expr = sprintf("at::%s(%s)", name, call_str),
         locals = locals, style = "function")
  }
}

gen_cpp_general <- function(parsed, variants = "") {
  n <- parsed$name
  ret <- parse_return_type(parsed$ret)
  is_inplace <- grepl("_$", n)
  is_method_only <- grepl("method", variants) && !grepl("function", variants)
  if (n %in% METHOD_ONLY_OPS) is_method_only <- TRUE

  # Filter out layout/pin_memory args (handled by TensorOptions)
  active_args <- Filter(function(a) !(a$name %in% c("layout", "pin_memory")),
                        parsed$args)

  # Build parameter list
  params <- vapply(active_args, function(a) {
    p <- cpp_param(a)
    if (is.null(p)) "" else p
  }, "")
  params <- params[nzchar(params)]
  param_str <- paste(params, collapse = ", ")

  # TensorOptions detection: replace dtype/device args with opts builder
  if (has_tensor_options(active_args)) {
    return(gen_cpp_tensor_options(parsed, active_args, ret, variants))
  }

  call <- build_call(n, active_args, variants)

  # In-place ops
  if (is_inplace) {
    body <- character()
    if (length(call$locals) > 0) body <- c(body, call$locals)
    body <- c(body, sprintf("    %s;", call$expr))
    body <- c(body, "    return self;")
    return(sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    ))
  }

  # Void return
  if (length(ret$types) == 1 && ret$types == "void") {
    body <- character()
    if (length(call$locals) > 0) body <- c(body, call$locals)
    body <- c(body, sprintf("    %s;", call$expr))
    body <- c(body, "    return R_NilValue;")
    return(sprintf(
      "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    ))
  }

  # Tensor[] return
  if (length(ret$types) == 1 && ret$types == "Tensor[]") {
    body <- character()
    if (length(call$locals) > 0) body <- c(body, call$locals)
    body <- c(body, sprintf("    auto result = %s;", call$expr))
    body <- c(body, "    return tensor_list_to_sexp(result);")
    return(sprintf(
      "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    ))
  }

  # Scalar return types (bool, int, float, Scalar, ScalarType, QScheme)
  if (length(ret$types) == 1 && ret$types %in% c("bool", "int", "float",
                                                    "Scalar", "ScalarType",
                                                    "QScheme")) {
    cpp_ret <- switch(ret$types,
      "bool" = "bool",
      "int" = "int64_t",
      "float" = "double",
      "Scalar" = "at::Scalar",
      "ScalarType" = "at::ScalarType",
      "QScheme" = "at::QScheme"
    )
    body <- character()
    if (length(call$locals) > 0) body <- c(body, call$locals)

    if (ret$types %in% c("ScalarType", "QScheme")) {
      # Return as integer (enum value)
      body <- c(body, sprintf("    return Rf_ScalarInteger(static_cast<int>(%s));",
                              call$expr))
      return(sprintf(
        "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
        n, param_str, paste(body, collapse = "\n")
      ))
    }

    if (ret$types == "Scalar") {
      # Return as SEXP via Rcpp::wrap
      body <- c(body, sprintf("    at::Scalar s = %s;", call$expr))
      body <- c(body, "    if (s.isIntegral(false)) return Rf_ScalarInteger(s.toLong());")
      body <- c(body, "    return Rf_ScalarReal(s.toDouble());")
      return(sprintf(
        "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
        n, param_str, paste(body, collapse = "\n")
      ))
    }

    if (length(call$locals) > 0) {
      body <- c(body, sprintf("    return %s;", call$expr))
      return(sprintf(
        "// [[Rcpp::export]]\n%s C_torch_%s(%s) {\n%s\n}",
        cpp_ret, n, param_str, paste(body, collapse = "\n")
      ))
    } else {
      return(sprintf(
        "// [[Rcpp::export]]\n%s C_torch_%s(%s) { return %s; }",
        cpp_ret, n, param_str, call$expr
      ))
    }
  }

  # Tuple return
  is_tuple <- length(ret$types) > 1
  if (is_tuple) {
    return(gen_cpp_tuple_return(n, param_str, call$expr, call$locals, ret))
  }

  # Simple Tensor return
  if (length(call$locals) > 0) {
    body <- c(
      call$locals,
      sprintf("    return %s;", call$expr)
    )
    sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    )
  } else {
    sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) { return %s; }",
      n, param_str, call$expr
    )
  }
}

# Generate C++ for tuple-returning ops
gen_cpp_tuple_return <- function(name, param_str, call_expr, locals, ret) {
  ntup <- length(ret$types)
  body <- character()
  if (length(locals) > 0) body <- c(body, locals)
  body <- c(body, sprintf("    auto result = %s;", call_expr))
  body <- c(body, sprintf("    SEXP out = PROTECT(Rf_allocVector(VECSXP, %d));",
                           ntup))
  for (i in seq_len(ntup)) {
    if (ret$types[i] == "Tensor[]") {
      body <- c(body, sprintf(
        "    SET_VECTOR_ELT(out, %d, tensor_list_to_sexp(std::get<%d>(result)));",
        i - 1L, i - 1L))
    } else {
      body <- c(body, sprintf(
        "    SET_VECTOR_ELT(out, %d, Rcpp::wrap(std::get<%d>(result)));",
        i - 1L, i - 1L))
    }
  }
  body <- c(body, sprintf("    SEXP names = PROTECT(Rf_allocVector(STRSXP, %d));",
                           ntup))
  for (i in seq_len(ntup)) {
    body <- c(body, sprintf('    SET_STRING_ELT(names, %d, Rf_mkChar("%s"));',
                            i - 1L, ret$names[i]))
  }
  body <- c(body, "    Rf_setAttrib(out, R_NamesSymbol, names);")
  body <- c(body, "    UNPROTECT(2);")
  body <- c(body, "    return out;")

  sprintf(
    "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
    name, param_str, paste(body, collapse = "\n")
  )
}

# Generate C++ for TensorOptions creation ops
gen_cpp_tensor_options <- function(parsed, active_args, ret, variants = "") {
  n <- parsed$name

  # Separate TensorOptions args from regular args
  opts_args <- c("dtype", "device", "layout", "pin_memory")
  # MemoryFormat is a trailing kwarg in _like ops, not part of TensorOptions
  has_memfmt <- any(vapply(active_args, function(a)
    a$name == "memory_format" && a$type == "MemoryFormat?", FALSE))
  skip_names <- opts_args
  if (has_memfmt) skip_names <- c(skip_names, "memory_format")
  regular_args <- Filter(function(a) !(a$name %in% skip_names), active_args)

  is_method_only <- grepl("method", variants) && !grepl("function", variants)
  if (n %in% METHOD_ONLY_OPS) is_method_only <- TRUE

  # For method-only ops, self is implicit — exclude from regular_args for call
  self_in_regular <- which(vapply(regular_args, function(a) a$name == "self", FALSE))

  # Build params: regular args + dtype + device (+ memory_format if present)
  params <- character()
  for (a in regular_args) {
    p <- cpp_param(a)
    if (!is.null(p) && nzchar(p)) params <- c(params, p)
  }
  params <- c(params, "SEXP dtype_sexp", "SEXP device_sexp")
  if (has_memfmt) params <- c(params, "SEXP memory_format")
  param_str <- paste(params, collapse = ", ")

  # Build body
  body <- character()

  # Locals for int[] etc in regular args (exclude self for method-only)
  args_for_locals <- if (is_method_only && length(self_in_regular) > 0) {
    regular_args[-self_in_regular[1]]
  } else {
    regular_args
  }
  body <- c(body, cpp_locals(args_for_locals))

  body <- c(body, "    auto opts = at::TensorOptions();")
  body <- c(body,
    "    auto dtype = sexp_to_dtype(dtype_sexp);",
    "    if (dtype.has_value()) opts = opts.dtype(dtype.value());",
    "    if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));"
  )

  # Build call args (excluding self for method-only)
  call_args_list <- if (is_method_only && length(self_in_regular) > 0) {
    regular_args[-self_in_regular[1]]
  } else {
    regular_args
  }
  call_parts <- character()
  for (a in call_args_list) {
    call_parts <- c(call_parts, cpp_call_expr(a))
  }
  call_parts <- c(call_parts, "opts")
  if (has_memfmt) {
    call_parts <- c(call_parts, "sexp_to_optional_memory_format(memory_format)")
  }
  call_str <- paste(call_parts, collapse = ", ")

  if (is_method_only) {
    body <- c(body, sprintf(
      "    return make_tensor_sexp(new at::Tensor(self.%s(%s)));",
      n, call_str))
  } else {
    body <- c(body, sprintf(
      "    return make_tensor_sexp(new at::Tensor(at::%s(%s)));",
      n, call_str))
  }

  sprintf(
    "// [[Rcpp::export]]\nSEXP C_torch_%s(%s) {\n%s\n}",
    n, param_str, paste(body, collapse = "\n")
  )
}

# ---- R code generation (general emitter) ----

# Translate Python default to R
translate_default <- function(d) {
  if (is.null(d)) return(NULL)
  d <- gsub("True", "TRUE", d)
  d <- gsub("False", "FALSE", d)
  d <- gsub("None", "NULL", d)
  # Handle list defaults like [0, 1] or []
  if (grepl("^\\[", d)) d <- "NULL"
  d
}

# Map ATen types to R-friendly doc descriptions
type_to_doc <- function(type) {
  type <- gsub("\\?$", "", type)  # strip optional marker
  switch(type,
    "Tensor"       = "A torch_tensor.",
    "Scalar"       = "A numeric scalar.",
    "ScalarType"   = "A torch dtype (e.g., torch_float32).",
    "int"          = "An integer.",
    "float"        = "A numeric value.",
    "bool"         = "A logical value.",
    "int[]"        = "An integer vector.",
    "float[]"      = "A numeric vector.",
    "Tensor[]"     = "A list of torch_tensors.",
    "Dimname"      = "A dimension name.",
    "Dimname[]"    = "A character vector of dimension names.",
    "MemoryFormat" = "A memory format.",
    "Device"       = "A device string (e.g., \"cpu\").",
    "Generator"    = "A random number generator.",
    "str"          = "A character string.",
    paste0("(", type, ").")
  )
}

# Generate roxygen block for a namespace function
gen_roxygen <- function(parsed) {
  n <- parsed$name
  title <- gsub("_$", " (in-place)", gsub("_", " ", n))
  title <- paste0(toupper(substring(title, 1, 1)), substring(title, 2))

  lines <- c(
    sprintf("#' %s", title),
    "#'",
    sprintf("#' Wrapper for \\code{at::%s}.", n)
  )

  # Filter out layout/pin_memory for param docs
  active_args <- Filter(function(a) !(a$name %in% c("layout", "pin_memory")),
                         parsed$args)
  for (a in active_args) {
    optional <- grepl("\\?$", a$type)
    desc <- type_to_doc(a$type)
    if (optional && !is.null(a$default)) desc <- paste0(desc, " Optional.")
    lines <- c(lines, sprintf("#' @param %s %s", a$name, desc))
  }

  lines <- c(lines,
    "#' @return A torch_tensor.",
    "#' @export"
  )
  paste(lines, collapse = "\n")
}

gen_r_general <- function(parsed, is_method = FALSE) {
  n <- parsed$name

  # Filter out layout/pin_memory
  active_args <- Filter(function(a) !(a$name %in% c("layout", "pin_memory")),
                        parsed$args)

  is_opts <- has_tensor_options(active_args)

  # For TensorOptions ops, further filter - we expose dtype + device only
  if (is_opts) {
    opts_skip <- c("layout", "pin_memory")
    active_args <- Filter(function(a) !(a$name %in% opts_skip), active_args)
  }

  # For methods, self is implicit — find and exclude it from formals
  if (is_method) {
    self_idx <- which(vapply(active_args, function(a) a$name == "self", FALSE))
    if (length(self_idx) == 0) self_idx <- 1L  # fallback: first arg is self
    r_mn <- if (grepl("_$", n)) sprintf('`%s`', n) else n
    if (length(active_args) <= 1 ||
        (length(active_args) == 1 && length(self_idx) == 1)) {
      return(sprintf('.tensor_methods$%s <- function(self) C_torch_%s(self)',
                     r_mn, n))
    }
    formal_args <- active_args[-self_idx[1]]
  } else {
    formal_args <- active_args
  }
  r_formals <- vapply(formal_args, function(a) {
    d <- translate_default(a$default)
    if (!is.null(d)) sprintf("%s = %s", a$name, d) else a$name
  }, "")

  # Build C call args
  c_args <- vapply(active_args, function(a) {
    type <- a$type
    name <- a$name
    if (type == "int") {
      if (name == "self") return("self")
      return(sprintf("as.integer(%s)", name))
    }
    if (type == "float") return(sprintf("as.double(%s)", name))
    if (type == "bool") return(sprintf("as.logical(%s)", name))
    # Everything else (Tensor, Scalar, SEXP types) pass through
    name
  }, "")

  c_call <- sprintf("C_torch_%s(%s)", n, paste(c_args, collapse = ", "))

  # R names with trailing _ need backtick quoting for $ access
  r_method_name <- if (grepl("_$", n)) sprintf('`%s`', n) else n

  if (is_method) {
    if (length(formal_args) == 0) {
      sprintf('.tensor_methods$%s <- function(self) %s', r_method_name, c_call)
    } else {
      sprintf('.tensor_methods$%s <- function(self, %s) {\n    %s\n}',
              r_method_name, paste(r_formals, collapse = ", "), c_call)
    }
  } else {
    sprintf('%s\ntorch_%s <- function(%s) {\n    %s\n}',
            gen_roxygen(parsed), n, paste(r_formals, collapse = ", "), c_call)
  }
}

# ---- Exclusion filters ----

# Only exclude ops that are truly internal or backend-specific
EXCLUDE_PATTERNS <- c(
  "backward",                    # autograd internals
  "^fbgemm",                     # Facebook GEMM backend
  "^mkldnn", "^to_mkldnn",      # Intel MKL-DNN backend
  "^cudnn",                      # cuDNN backend
  "^miopen",                     # AMD MIOpen backend
  "^_[a-z]",                     # internal ops (leading underscore)
  "_copy$",                      # copy variants (internal dispatch)
  "^lift$", "^lift_fresh",       # autograd internals
  "^infinitely_differentiable_gelu"  # internal gelu variant
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

  if (is.null(existing_ops)) {
    existing_ops <- scan_existing_ops()
  }

  cpp_lines <- character()
  r_ns_lines <- character()
  r_method_lines <- character()
  generated <- character()
  excluded <- character()
  skipped_types <- list()
  failed_ops <- character()

  for (entry in entries) {
    if (entry$manual) next

    parsed <- parse_sig(entry$sig)
    if (is.null(parsed)) next

    # Apply exclusion filters
    if (should_exclude(parsed$name)) {
      excluded <- c(excluded, parsed$name)
      next
    }

    # Skip out= variants (by overload name or by structural detection)
    if (grepl("out$", parsed$overload) || entry$is_out) next

    # Skip if we already have this op hand-written
    if (parsed$name %in% existing_ops) next

    # Skip overloads — only take primary (empty overload) or first seen
    if (parsed$name %in% generated) next

    if (!can_generate(parsed)) {
      # Track what types we can't handle yet
      for (a in parsed$args) {
        if (!is_supported_arg(a$type) && !(a$type %in% SKIP_ARG_TYPES) &&
            !(a$name %in% c("layout", "pin_memory"))) {
          skipped_types[[a$type]] <- (skipped_types[[a$type]] %||% 0L) + 1L
        }
      }
      ret <- parse_return_type(parsed$ret)
      if (is.null(ret) || !is_supported_return(ret)) {
        rkey <- paste0("ret:", parsed$ret)
        skipped_types[[rkey]] <- (skipped_types[[rkey]] %||% 0L) + 1L
      }
      next
    }

    # Generate C++
    cpp <- tryCatch(
      gen_cpp_general(parsed, variants = entry$variants),
      error = function(e) {
        failed_ops <<- c(failed_ops, sprintf("%s: %s", parsed$name, e$message))
        NULL
      }
    )
    if (is.null(cpp)) next
    cpp_lines <- c(cpp_lines, cpp)

    # Generate R namespace function
    r_ns <- gen_r_general(parsed, is_method = FALSE)
    r_ns_lines <- c(r_ns_lines, r_ns)

    # Generate R method if variants include "method"
    if (grepl("method", entry$variants)) {
      r_method <- gen_r_general(parsed, is_method = TRUE)
      r_method_lines <- c(r_method_lines, r_method)
    }

    generated <- c(generated, parsed$name)
  }

  cat("\nGenerated", length(generated), "ops\n")
  cat("Hand-written:", length(existing_ops), "ops\n")
  cat("Excluded:", length(unique(excluded)), "ops\n")

  if (length(failed_ops) > 0) {
    cat("\nFailed to generate:\n")
    for (f in failed_ops) cat("  ", f, "\n")
  }

  if (length(skipped_types) > 0) {
    cat("\nUnsupported arg/return types (op count):\n")
    sorted <- sort(unlist(skipped_types), decreasing = TRUE)
    for (nm in names(sorted)) {
      cat(sprintf("  %-30s %d ops\n", nm, sorted[nm]))
    }
  }

  if (dry_run) {
    cat("\n=== DRY RUN ===\n")
    return(invisible(list(
      ops = generated,
      excluded = unique(excluded),
      cpp = cpp_lines,
      r_namespace = r_ns_lines,
      r_methods = r_method_lines,
      skipped_types = skipped_types
    )))
  }

  # Write C++ file
  cpp_header <- paste0(
    "// Auto-generated by tools/codegen.R -- DO NOT EDIT\n",
    "// Source: native_functions.yaml\n",
    "#include \"tinytorch.h\"\n\n"
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
  exports_cpp <- readLines("src/RcppExports.cpp")
  exports_cpp <- sub("#include <Rcpp.h>", '#include "tinytorch.h"', exports_cpp,
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
    if (basename(f) %in% c("gen-ops.cpp", "RcppExports.cpp")) next
    lines <- readLines(f)
    m <- regmatches(lines, gregexpr("C_torch_(\\w+)", lines))
    ops <- c(ops, unlist(lapply(m, function(x) sub("C_torch_", "", x))))
  }

  # Scan R for C_torch_* calls
  r_files <- list.files(r_dir, pattern = "\\.R$", full.names = TRUE)
  for (f in r_files) {
    if (basename(f) %in% c("zzz-gen-ops.R", "RcppExports.R")) next
    lines <- readLines(f)
    m <- regmatches(lines, gregexpr("C_torch_(\\w+)", lines))
    ops <- c(ops, unlist(lapply(m, function(x) sub("C_torch_", "", x))))
    # Also scan for hand-written torch_* function definitions
    # (may use C_nnf_* or other C function names)
    m2 <- regmatches(lines, gregexpr("^torch_(\\w+)\\s*<-\\s*function", lines))
    ops <- c(ops, unlist(lapply(m2, function(x) sub("^torch_(\\w+)\\s.*", "\\1", x))))
    # Also scan for hand-written method table entries: .tensor_methods$name <- function
    m3 <- regmatches(lines,
      gregexpr("\\.tensor_methods\\$`?(\\w+)`?\\s*<-\\s*function", lines))
    for (match in m3) {
      if (length(match) > 0) {
        nm <- sub(".*\\$`?(\\w+)`?\\s.*", "\\1", match)
        ops <- c(ops, nm)
      }
    }
  }

  unique(ops)
}

# If run directly
if (!interactive() && identical(sys.nframe(), 0L)) {
  result <- codegen(dry_run = TRUE)
}
