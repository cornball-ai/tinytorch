# ---- Namespace-level functions ----

#' Add two tensors
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @param alpha Scalar multiplier for other (default 1).
#' @export
torch_add <- function(self, other, alpha = 1) {
  .Call(C_torch_add, self, other, alpha)
}

#' Subtract two tensors
#' @export
torch_sub <- function(self, other, alpha = 1) {
  .Call(C_torch_sub, self, other, alpha)
}

#' Multiply two tensors element-wise
#' @export
torch_mul <- function(self, other) {
  .Call(C_torch_mul, self, other)
}

#' Divide two tensors element-wise
#' @export
torch_div <- function(self, other) {
  .Call(C_torch_div, self, other)
}

#' Matrix multiplication
#' @export
torch_matmul <- function(self, other) {
  .Call(C_torch_matmul, self, other)
}

#' Matrix-matrix multiplication (2D only)
#' @export
torch_mm <- function(self, other) {
  .Call(C_torch_mm, self, other)
}

#' Sum of tensor elements
#' @param self A torch_tensor.
#' @param dim Optional dimension to reduce (1-based).
#' @param keepdim Whether to keep the reduced dimension.
#' @export
torch_sum <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_sum, self, dim, keepdim)
}

#' Mean of tensor elements
#' @export
torch_mean <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_mean, self, dim, keepdim)
}

#' Sigmoid activation (namespace-level)
#' @export
torch_sigmoid <- function(self) {
  .Call(C_torch_sigmoid, self)
}

# ---- Method dispatch table ----

.tensor_methods <- new.env(parent = emptyenv())
.tensor_properties <- new.env(parent = emptyenv())

# Methods (return bound closures via $)
.tensor_methods$add <- function(self, other, alpha = 1) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_add_scalar, self, other)
  } else {
    .Call(C_torch_add, self, other, alpha)
  }
}

.tensor_methods$sub <- function(self, other, alpha = 1) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_sub_scalar, self, other)
  } else {
    .Call(C_torch_sub, self, other, alpha)
  }
}

.tensor_methods$mul <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_mul_scalar, self, other)
  } else {
    .Call(C_torch_mul, self, other)
  }
}

.tensor_methods$div <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_div_scalar, self, other)
  } else {
    .Call(C_torch_div, self, other)
  }
}

.tensor_methods$neg <- function(self) {
  .Call(C_torch_neg, self)
}

.tensor_methods$matmul <- function(self, other) {
  .Call(C_torch_matmul, self, other)
}

.tensor_methods$mm <- function(self, other) {
  .Call(C_torch_mm, self, other)
}

.tensor_methods$t <- function(self) {
  .Call(C_torch_t, self)
}

.tensor_methods$sum <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_sum, self, dim, keepdim)
}

.tensor_methods$mean <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_mean, self, dim, keepdim)
}

.tensor_methods$max <- function(self, dim = NULL) {
  .Call(C_torch_max, self, dim)
}

.tensor_methods$min <- function(self, dim = NULL) {
  .Call(C_torch_min, self, dim)
}

.tensor_methods$reshape <- function(self, shape) {
  .Call(C_torch_reshape, self, as.integer(shape))
}

.tensor_methods$view <- function(self, shape) {
  .Call(C_torch_view, self, as.integer(shape))
}

.tensor_methods$squeeze <- function(self, dim = NULL) {
  .Call(C_torch_squeeze, self, dim)
}

.tensor_methods$unsqueeze <- function(self, dim) {
  .Call(C_torch_unsqueeze, self, as.integer(dim))
}

.tensor_methods$clone <- function(self) {
  .Call(C_torch_clone, self)
}

.tensor_methods$contiguous <- function(self) {
  .Call(C_torch_contiguous, self)
}

.tensor_methods$to <- function(self, dtype) {
  .Call(C_torch_to_dtype, self, unclass(dtype))
}

.tensor_methods$item <- function(self) {
  .Call(C_torch_item, self)
}

.tensor_methods$size <- function(self, dim = NULL) {
  s <- .Call(C_tensor_shape, self)
  if (!is.null(dim)) s[dim] else s
}

# Unary methods
.tensor_methods$relu <- function(self) {
  .Call(C_torch_relu, self)
}

.tensor_methods$sigmoid <- function(self) {
  .Call(C_torch_sigmoid, self)
}

.tensor_methods$tanh <- function(self) {
  .Call(C_torch_tanh, self)
}

.tensor_methods$exp <- function(self) {
  .Call(C_torch_exp, self)
}

.tensor_methods$log <- function(self) {
  .Call(C_torch_log, self)
}

.tensor_methods$log2 <- function(self) {
  .Call(C_torch_log2, self)
}

.tensor_methods$log10 <- function(self) {
  .Call(C_torch_log10, self)
}

.tensor_methods$sqrt <- function(self) {
  .Call(C_torch_sqrt, self)
}

.tensor_methods$abs <- function(self) {
  .Call(C_torch_abs, self)
}

.tensor_methods$sign <- function(self) {
  .Call(C_torch_sign, self)
}

.tensor_methods$floor <- function(self) {
  .Call(C_torch_floor, self)
}

.tensor_methods$ceil <- function(self) {
  .Call(C_torch_ceil, self)
}

.tensor_methods$round <- function(self) {
  .Call(C_torch_round, self)
}

.tensor_methods$trunc <- function(self) {
  .Call(C_torch_trunc, self)
}

.tensor_methods$detach <- function(self) {
  .Call(C_torch_detach, self)
}

.tensor_methods$sin <- function(self) {
  .Call(C_torch_sin, self)
}

.tensor_methods$cos <- function(self) {
  .Call(C_torch_cos, self)
}

.tensor_methods$rsqrt <- function(self) {
  .Call(C_torch_rsqrt, self)
}

.tensor_methods$softmax <- function(self, dim) {
  .Call(C_nnf_softmax, self, as.integer(dim))
}

.tensor_methods$requires_grad_ <- function(self, requires_grad = TRUE) {
  # No-op for now (no autograd support)
  invisible(self)
}

# No-ops for CPU-only (return self unchanged)
.tensor_methods$cpu <- function(self) self

# Binary methods
.tensor_methods$pow <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_pow_scalar, self, other)
  } else {
    .Call(C_torch_pow, self, other)
  }
}

.tensor_methods$remainder <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_remainder_scalar, self, other)
  } else {
    .Call(C_torch_remainder, self, other)
  }
}

.tensor_methods$floor_divide <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_floor_divide_scalar, self, other)
  } else {
    .Call(C_torch_floor_divide, self, other)
  }
}

# Comparison methods
.tensor_methods$eq <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_eq_scalar, self, other)
  } else {
    .Call(C_torch_eq, self, other)
  }
}

.tensor_methods$ne <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_ne_scalar, self, other)
  } else {
    .Call(C_torch_ne, self, other)
  }
}

.tensor_methods$lt <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_lt_scalar, self, other)
  } else {
    .Call(C_torch_lt, self, other)
  }
}

.tensor_methods$le <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_le_scalar, self, other)
  } else {
    .Call(C_torch_le, self, other)
  }
}

.tensor_methods$gt <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_gt_scalar, self, other)
  } else {
    .Call(C_torch_gt, self, other)
  }
}

.tensor_methods$ge <- function(self, other) {
  if (is.numeric(other) && !inherits(other, "torch_tensor")) {
    .Call(C_torch_ge_scalar, self, other)
  } else {
    .Call(C_torch_ge, self, other)
  }
}

# Additional linalg/shape methods
.tensor_methods$bmm <- function(self, other) {
  .Call(C_torch_bmm, self, other)
}

.tensor_methods$transpose <- function(self, dim0, dim1) {
  .Call(C_torch_transpose, self, as.integer(dim0), as.integer(dim1))
}

.tensor_methods$flatten <- function(self, start_dim = 1L, end_dim = -1L) {
  .Call(C_torch_flatten, self, as.integer(start_dim), as.integer(end_dim))
}

.tensor_methods$dim <- function(self) {
  .Call(C_tensor_ndim, self)
}

.tensor_methods$size <- function(self, dim = NULL) {
  if (is.null(dim)) return(.Call(C_tensor_shape, self))
  .Call(C_tensor_shape, self)[dim]
}

# Properties (no extra args beyond self)
.tensor_properties$shape <- function(self) {
  .Call(C_tensor_shape, self)
}

.tensor_properties$dtype <- function(self) {
  code <- .Call(C_tensor_dtype, self)
  structure(code, class = "torch_dtype")
}

.tensor_properties$device <- function(self) {
  .Call(C_tensor_device, self)
}

.tensor_properties$ndim <- function(self) {
  .Call(C_tensor_ndim, self)
}

.tensor_properties$requires_grad <- function(self) {
  .Call(C_tensor_requires_grad, self)
}

.tensor_properties$is_cuda <- function(self) {
  FALSE
}

# ---- $ dispatch ----

#' @export
`$.torch_tensor` <- function(x, name) {
  fn <- .tensor_methods[[name]]
  if (!is.null(fn)) {
    self <- x
    return(function(...) fn(self, ...))
  }
  prop <- .tensor_properties[[name]]
  if (!is.null(prop)) {
    return(prop(x))
  }
  stop(sprintf("no member '%s' on torch_tensor", name))
}

# ---- print ----

#' @export
print.torch_tensor <- function(x, ...) {
  .Call(C_tensor_print, x)
  shape <- .Call(C_tensor_shape, x)
  dtype_code <- .Call(C_tensor_dtype, x)
  cat(sprintf("[ Rtorch — %s [ %s ] ]\n",
              .dtype_name(dtype_code),
              paste(shape, collapse = ", ")))
  invisible(x)
}

# ---- S3 operators ----

#' @export
`+.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) return(e1)
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_add, e1, e2, 1)
  } else {
    .Call(C_torch_add_scalar, e1, e2)
  }
}

#' @export
`-.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) return(.Call(C_torch_neg, e1))
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_sub, e1, e2, 1)
  } else {
    .Call(C_torch_sub_scalar, e1, e2)
  }
}

#' @export
`*.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_mul, e1, e2)
  } else {
    .Call(C_torch_mul_scalar, e1, e2)
  }
}

#' @export
`/.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_div, e1, e2)
  } else {
    .Call(C_torch_div_scalar, e1, e2)
  }
}

#' @export
`^.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_pow, e1, e2)
  } else {
    .Call(C_torch_pow_scalar, e1, e2)
  }
}

#' @export
`%%.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_remainder, e1, e2)
  } else {
    .Call(C_torch_remainder_scalar, e1, e2)
  }
}

#' @export
`%/%.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_floor_divide, e1, e2)
  } else {
    .Call(C_torch_floor_divide_scalar, e1, e2)
  }
}

#' @export
`==.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_eq, e1, e2)
  } else {
    .Call(C_torch_eq_scalar, e1, e2)
  }
}

#' @export
`!=.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_ne, e1, e2)
  } else {
    .Call(C_torch_ne_scalar, e1, e2)
  }
}

#' @export
`<.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_lt, e1, e2)
  } else {
    .Call(C_torch_lt_scalar, e1, e2)
  }
}

#' @export
`<=.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_le, e1, e2)
  } else {
    .Call(C_torch_le_scalar, e1, e2)
  }
}

#' @export
`>.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_gt, e1, e2)
  } else {
    .Call(C_torch_gt_scalar, e1, e2)
  }
}

#' @export
`>=.torch_tensor` <- function(e1, e2) {
  if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_ge, e1, e2)
  } else {
    .Call(C_torch_ge_scalar, e1, e2)
  }
}
