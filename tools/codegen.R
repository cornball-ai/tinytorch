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

    # Detect out-variant: non-self args with Tensor(a!) aliasing
    # Match Tensor(letter!) but not the first Tensor(a!) which is self for in-place
    is_out <- is_out_variant(sig)

    entries[[i]] <- list(sig = sig, variants = variants, manual = manual,
                         tags = tags, is_out = is_out)
  }

  entries
}

# Detect out-variant signatures by looking for output tensor args.
# Out variants have Tensor(a!) args that are NOT self (self is the first arg
# in in-place ops). The pattern: if the sig has Tensor(letter!) args after
# the keyword-only marker (*), those are output args.
is_out_variant <- function(sig) {
  # Get args part
  m <- regmatches(sig, regexec("\\((.*)\\)", sig))[[1]]
  if (length(m) < 2) return(FALSE)
  args_str <- m[2]

  # Check for *, Tensor(a!) pattern (output args after keyword-only marker)
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
#   needs_local: if TRUE, emitter generates a local variable (multi-line body)

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
  "ScalarType" = list(
    cpp_type = "SEXP",
    cpp_call = "sexp_to_dtype(%s).value()",
    r_convert = NULL
  )
)

# int[] patterns (int[], int[1], int[2], int[3], etc.) and optional variants
is_int_array_type <- function(type) {
  grepl("^int\\[\\d*\\]\\??$", type)
}

# Tensor[] pattern
is_tensor_list_type <- function(type) {
  type == "Tensor[]"
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

  # Tuple: (Tensor foo, Tensor bar)
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

  # Simple types
  if (ret_str %in% c("Tensor", "bool", "int", "float", "Scalar")) {
    return(list(types = ret_str, names = NULL))
  }

  NULL
}

# ---- Generability check ----

# Is this argument type supported by codegen?
is_supported_arg <- function(type) {
  if (type %in% names(ARG_CONVERSIONS)) return(TRUE)
  if (is_int_array_type(type)) return(TRUE)
  if (is_tensor_list_type(type)) return(TRUE)
  FALSE
}

# Is this return type supported?
is_supported_return <- function(ret) {
  if (is.null(ret)) return(FALSE)
  # Simple Tensor return
  if (length(ret$types) == 1 && ret$types == "Tensor") return(TRUE)
  # Tuple of Tensors
  if (length(ret$types) > 1 && all(ret$types == "Tensor")) return(TRUE)
  FALSE
}

# TensorOptions quartet detection
TENSOR_OPTIONS_NAMES <- c("dtype", "layout", "device", "pin_memory")

has_tensor_options <- function(args) {
  arg_names <- vapply(args, function(a) a$name, "")
  all(c("dtype", "device") %in% arg_names) &&
    any(vapply(args, function(a) a$type == "ScalarType?", FALSE))
}

# Filter: skip types we can't generate yet
SKIP_ARG_TYPES <- c("Layout?", "MemoryFormat?", "Generator?", "Dimname[]?",
                     "Dimname[]", "Scalar[]", "Tensor?[]", "Storage", "Stream",
                     "float[]?")

can_generate <- function(parsed) {
  if (is.null(parsed)) return(FALSE)

  ret <- parse_return_type(parsed$ret)
  if (!is_supported_return(ret)) return(FALSE)

  for (a in parsed$args) {
    # Skip the TensorOptions args we'll handle specially
    if (a$name %in% c("layout", "pin_memory") &&
        a$type %in% c("Layout?", "bool?")) next
    # Skip MemoryFormat?, Generator?, Dimname
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

  if (is_int_array_type(type)) {
    return(sprintf("SEXP %s_sexp", name))
  }

  if (is_tensor_list_type(type)) {
    return(sprintf("SEXP %s_sexp", name))
  }

  NULL
}

# Get the C++ call expression for an arg
cpp_call_expr <- function(a) {
  type <- a$type
  name <- a$name

  if (name %in% c("layout", "pin_memory")) return(NULL)

  if (type %in% names(ARG_CONVERSIONS)) {
    conv <- ARG_CONVERSIONS[[type]]
    if (!is.null(conv$cpp_call)) {
      actual_name <- if (type == "Scalar") paste0(name, "_sexp") else name
      return(sprintf(conv$cpp_call, actual_name))
    }
    return(name)
  }

  if (is_int_array_type(type)) {
    is_optional <- grepl("\\?$", type)
    if (is_optional) {
      return(sprintf("%s_ref", name))
    }
    return(sprintf("at::IntArrayRef(%s_vec.data(), %s_vec.size())",
                   name, name))
  }

  if (is_tensor_list_type(type)) {
    return(sprintf("%s_vec", name))
  }

  name
}

# Generate locals needed before the call (int[] vectors, Tensor[] lists)
cpp_locals <- function(args) {
  lines <- character()
  for (a in args) {
    if (is_int_array_type(a$type)) {
      is_optional <- grepl("\\?$", a$type)
      if (is_optional) {
        # Optional int array: convert to vec only if non-NULL
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
    if (is_tensor_list_type(a$type)) {
      lines <- c(lines, sprintf("    auto %s_vec = sexp_to_tensor_list(%s_sexp);",
                                a$name, a$name))
    }
  }
  lines
}

# Does this op need a multi-line body?
needs_multiline <- function(parsed) {
  ret <- parse_return_type(parsed$ret)
  if (length(ret$types) > 1) return(TRUE)  # tuple return
  for (a in parsed$args) {
    if (is_int_array_type(a$type)) return(TRUE)
    if (is_tensor_list_type(a$type)) return(TRUE)
    if (has_tensor_options(parsed$args) &&
        a$name == "dtype" && a$type == "ScalarType?") return(TRUE)
  }
  FALSE
}

gen_cpp_general <- function(parsed, variants = "") {
  n <- parsed$name
  ret <- parse_return_type(parsed$ret)
  is_inplace <- grepl("_$", n)

  # Filter out layout/pin_memory args
  active_args <- Filter(function(a) !(a$name %in% c("layout", "pin_memory")),
                        parsed$args)

  # Build parameter list
  params <- vapply(active_args, function(a) {
    p <- cpp_param(a)
    if (is.null(p)) "" else p
  }, "")
  params <- params[nzchar(params)]
  param_str <- paste(params, collapse = ", ")

  # In-place ops: use method style if variants says "method" (no at:: free fn),
  # otherwise use at:: free function style.
  if (is_inplace) {
    is_method_only <- grepl("method", variants) && !grepl("function", variants)

    if (is_method_only) {
      # self.op_(arg2, arg3, ...) — method call, skip self in call args
      self_idx <- which(vapply(active_args, function(a) a$name == "self", FALSE))
      if (length(self_idx) == 0) self_idx <- 1L
      non_self_args <- active_args[-self_idx[1]]
      call_parts <- vapply(non_self_args, function(a) {
        e <- cpp_call_expr(a)
        if (is.null(e)) "" else e
      }, "")
      call_parts <- call_parts[nzchar(call_parts)]
      call_str <- paste(call_parts, collapse = ", ")
      locals <- cpp_locals(non_self_args)

      body <- character()
      if (length(locals) > 0) body <- c(body, locals)
      body <- c(body, sprintf("    self.%s(%s);", n, call_str))
      body <- c(body, "    return self;")
    } else {
      # at::op_(self, ...) — free function call
      call_parts <- vapply(active_args, function(a) {
        e <- cpp_call_expr(a)
        if (is.null(e)) "" else e
      }, "")
      call_parts <- call_parts[nzchar(call_parts)]
      call_str <- paste(call_parts, collapse = ", ")
      locals <- cpp_locals(active_args)

      body <- character()
      if (length(locals) > 0) body <- c(body, locals)
      body <- c(body, sprintf("    at::%s(%s);", n, call_str))
      body <- c(body, "    return self;")
    }

    return(sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    ))
  }

  # Build call argument list (non-inplace)
  call_parts <- vapply(active_args, function(a) {
    e <- cpp_call_expr(a)
    if (is.null(e)) "" else e
  }, "")
  call_parts <- call_parts[nzchar(call_parts)]
  call_str <- paste(call_parts, collapse = ", ")

  locals <- cpp_locals(active_args)
  is_tuple <- length(ret$types) > 1

  # TensorOptions detection: replace dtype/device args with opts builder
  if (has_tensor_options(active_args)) {
    return(gen_cpp_tensor_options(parsed, active_args, ret))
  }

  if (is_tuple) {
    return(gen_cpp_tuple_return(n, param_str, call_str, locals, ret))
  }

  if (length(locals) > 0) {
    body <- c(
      locals,
      sprintf("    return at::%s(%s);", n, call_str)
    )
    sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) {\n%s\n}",
      n, param_str, paste(body, collapse = "\n")
    )
  } else {
    sprintf(
      "// [[Rcpp::export]]\nat::Tensor C_torch_%s(%s) { return at::%s(%s); }",
      n, param_str, n, call_str
    )
  }
}

# Generate C++ for tuple-returning ops
gen_cpp_tuple_return <- function(name, param_str, call_str, locals, ret) {
  ntup <- length(ret$types)
  body <- character()
  if (length(locals) > 0) body <- c(body, locals)
  body <- c(body, sprintf("    auto result = at::%s(%s);", name, call_str))
  body <- c(body, sprintf("    SEXP out = PROTECT(Rf_allocVector(VECSXP, %d));",
                           ntup))
  for (i in seq_len(ntup)) {
    body <- c(body, sprintf(
      "    SET_VECTOR_ELT(out, %d, Rcpp::wrap(std::get<%d>(result)));",
      i - 1L, i - 1L))
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
gen_cpp_tensor_options <- function(parsed, active_args, ret) {
  n <- parsed$name

  # Separate TensorOptions args from regular args
  opts_args <- c("dtype", "device", "layout", "pin_memory")
  regular_args <- Filter(function(a) !(a$name %in% opts_args), active_args)
  has_device <- any(vapply(active_args, function(a) a$name == "device", FALSE))

  # Build params: regular args + dtype + device
  params <- character()
  for (a in regular_args) {
    p <- cpp_param(a)
    if (!is.null(p) && nzchar(p)) params <- c(params, p)
  }
  params <- c(params, "SEXP dtype_sexp", "SEXP device_sexp")
  param_str <- paste(params, collapse = ", ")

  # Build body
  body <- character()

  # Locals for int[] etc in regular args
  for (a in regular_args) {
    if (is_int_array_type(a$type)) {
      body <- c(body, sprintf("    auto %s_vec = sexp_to_int_vec(%s_sexp);",
                              a$name, a$name))
    }
    if (is_tensor_list_type(a$type)) {
      body <- c(body, sprintf("    auto %s_vec = sexp_to_tensor_list(%s_sexp);",
                              a$name, a$name))
    }
  }

  body <- c(body, "    auto opts = at::TensorOptions();")
  body <- c(body,
    "    auto dtype = sexp_to_dtype(dtype_sexp);",
    "    if (dtype.has_value()) opts = opts.dtype(dtype.value());",
    "    if (!Rf_isNull(device_sexp)) opts = opts.device(sexp_to_device(device_sexp));"
  )

  # Build call args
  call_parts <- character()
  for (a in regular_args) {
    call_parts <- c(call_parts, cpp_call_expr(a))
  }
  call_parts <- c(call_parts, "opts")
  call_str <- paste(call_parts, collapse = ", ")

  body <- c(body, sprintf(
    "    return make_tensor_sexp(new at::Tensor(at::%s(%s)));",
    n, call_str))

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

gen_r_general <- function(parsed, is_method = FALSE) {
  n <- parsed$name

  # Filter out layout/pin_memory
  active_args <- Filter(function(a) !(a$name %in% c("layout", "pin_memory")),
                        parsed$args)

  is_opts <- has_tensor_options(active_args)

  # For TensorOptions ops, further filter - we expose dtype + device only
  if (is_opts) {
    # Keep non-options args, plus dtype and device
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
    sprintf('#\' @export\ntorch_%s <- function(%s) {\n    %s\n}',
            n, paste(r_formals, collapse = ", "), c_call)
  }
}

# ---- Exclusion filters ----

EXCLUDE_PATTERNS <- c(
  "backward",
  "^fbgemm", "^mkldnn", "^cudnn", "^miopen",
  "quantize", "^fake_quantize",
  "^q_scale$", "^q_zero_point$", "^q_per_channel",
  "^int_repr$", "^dequantize$",
  "sparse",
  "^coalesce$", "^is_coalesced$",
  "^indices$", "^values$",
  "^crow_indices$", "^col_indices$", "^ccol_indices$", "^row_indices$",
  "_copy$",
  "^lift", "^lift_fresh",
  "^is_distributed$",
  "^is_same_size$",
  "^norm_except_dim$",
  "^native_norm$",
  "^nuclear_norm$",
  "^hspmm$", "^smm$",
  "^to_sparse$", "^to_mkldnn",
  "^dense_dim$", "^sparse_dim$",
  "^align_as$", "^align_to",
  "^numpy_T$",
  "^matrix_H$", "^mT$", "^mH$",
  "^resolve_conj$", "^resolve_neg$", "^conj_physical$",
  "^one_hot$",
  "^type_as$",
  "^is_set_to$",
  "^infinitely_differentiable_gelu",
  "^ldexp$",
  "^matrix_exp_backward$",
  "^combinations$",
  "^heaviside$",
  "^expand_as$", "^reshape_as$", "^view_as$",
  "^unfold$",
  "^equal$",
  "^is_nonzero$",
  "^sum_to_size$",        # method-only, not in at::
  "^to_dense$",           # method-only, not in at::
  "^new_empty$", "^new_empty_strided$", "^new_full$", "^new_zeros$", "^new_ones$",
  "^pin_memory$",         # method-only
  "^to_padded_tensor$",   # method-only
  "^_[a-z]"  # internal ops (leading underscore)
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
      next
    }

    # Generate C++
    cpp <- gen_cpp_general(parsed, variants = entry$variants)
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

  if (length(skipped_types) > 0) {
    cat("\nUnsupported arg types (op count):\n")
    sorted <- sort(unlist(skipped_types), decreasing = TRUE)
    for (nm in names(sorted)) {
      cat(sprintf("  %-20s %d ops\n", nm, sorted[nm]))
    }
  }

  if (dry_run) {
    cat("\n=== DRY RUN ===\n")
    cat("Would generate:", paste(generated, collapse = ", "), "\n")
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
    if (basename(f) == "gen-ops.cpp") next
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
    # Also scan for hand-written torch_* function definitions
    # (may use C_nnf_* or other C function names)
    m2 <- regmatches(lines, gregexpr("^torch_(\\w+)\\s*<-\\s*function", lines))
    ops <- c(ops, unlist(lapply(m2, function(x) sub("^torch_(\\w+)\\s.*", "\\1", x))))
  }

  unique(ops)
}

# If run directly
if (!interactive() && identical(sys.nframe(), 0L)) {
  result <- codegen(dry_run = TRUE)
}
