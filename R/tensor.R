# ---- Namespace-level functions ----

#' Add two tensors
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @param alpha Scalar multiplier for other (default 1).
#' @examples
#' \donttest{
#' a <- torch_ones(c(2, 3))
#' b <- torch_ones(c(2, 3))
#' torch_add(a, b)
#' }
#' @export
torch_add <- function(self, other, alpha = 1) {
  .Call(C_torch_add, self, other, alpha)
}

#' Subtract two tensors
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @param alpha Scalar multiplier for other (default 1).
#' @examples
#' \donttest{
#' a <- torch_ones(c(2, 3))
#' b <- torch_ones(c(2, 3))
#' torch_sub(a, b)
#' }
#' @export
torch_sub <- function(self, other, alpha = 1) {
  .Call(C_torch_sub, self, other, alpha)
}

#' Multiply two tensors element-wise
#' @param self A torch_tensor.
#' @param other A torch_tensor or scalar.
#' @examples
#' \donttest{
#' a <- torch_randn(c(2, 3))
#' torch_mul(a, 2.0)
#' }
#' @export
torch_mul <- function(self, other) {
  .Call(C_torch_mul, self, other)
}

#' Divide two tensors element-wise
#' @param self A torch_tensor.
#' @param other A torch_tensor or scalar.
#' @examples
#' \donttest{
#' a <- torch_randn(c(2, 3))
#' torch_div(a, 2.0)
#' }
#' @export
torch_div <- function(self, other) {
  .Call(C_torch_div, self, other)
}

#' Matrix multiplication
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @examples
#' \donttest{
#' a <- torch_randn(c(2, 3))
#' b <- torch_randn(c(3, 4))
#' torch_matmul(a, b)
#' }
#' @export
torch_matmul <- function(self, other) {
  .Call(C_torch_matmul, self, other)
}

#' Matrix-matrix multiplication (2D only)
#' @param self A torch_tensor.
#' @param other A torch_tensor.
#' @examples
#' \donttest{
#' a <- torch_randn(c(2, 3))
#' b <- torch_randn(c(3, 4))
#' torch_mm(a, b)
#' }
#' @export
torch_mm <- function(self, other) {
  .Call(C_torch_mm, self, other)
}

#' Sum of tensor elements
#' @param self A torch_tensor.
#' @param dim Optional dimension to reduce (1-based).
#' @param keepdim Whether to keep the reduced dimension.
#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 3))
#' torch_sum(x)
#' torch_sum(x, dim = 1)
#' }
#' @export
torch_sum <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_sum, self, dim, keepdim)
}

#' Mean of tensor elements
#' @param self A torch_tensor.
#' @param dim Optional dimension to reduce (1-based).
#' @param keepdim Whether to keep the reduced dimension.
#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 3))
#' torch_mean(x)
#' torch_mean(x, dim = 1)
#' }
#' @export
torch_mean <- function(self, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_mean, self, dim, keepdim)
}

#' Sigmoid activation (namespace-level)
#' @param self A torch_tensor.
#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 3))
#' torch_sigmoid(x)
#' }
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

.tensor_methods$to <- function(self, dtype = NULL, device = NULL) {
  if (!is.null(dtype)) {
    .Call(C_torch_to_dtype, self, unclass(dtype))
  } else {
    self  # device is a no-op (CPU only)
  }
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
.tensor_methods$cuda <- function(self, ...) self
.tensor_methods$float <- function(self) .Call(C_torch_to_dtype, self, 6L)  # torch_float32
.tensor_methods$half <- function(self) .Call(C_torch_to_dtype, self, 5L)  # torch_float16
.tensor_methods$long <- function(self) .Call(C_torch_to_dtype, self, 4L)  # torch_int64

# Permute dimensions
.tensor_methods$permute <- function(self, dims) {
  .Call(C_torch_permute, self, as.integer(dims))
}

# Expand to larger size
.tensor_methods$expand <- function(self, size) {
  .Call(C_torch_expand, self, as.integer(size))
}

# Gather elements along a dimension
.tensor_methods$gather <- function(self, dim, index) {
  .Call(C_torch_gather, self, as.integer(dim), index)
}

# Masked fill
.tensor_methods$masked_fill <- function(self, mask, value) {
  .Call(C_torch_masked_fill, self, mask, as.double(value))
}

.tensor_methods$masked_fill_ <- function(self, mask, value) {
  .Call(C_torch_masked_fill_, self, mask, as.double(value))
}

# Copy from another tensor (in-place)
.tensor_methods$copy_ <- function(self, src) {
  .Call(C_torch_copy_, self, src)
}

# Norm
.tensor_methods$norm <- function(self, p = 2, dim = NULL, keepdim = FALSE) {
  .Call(C_torch_norm, self, as.double(p), dim, as.logical(keepdim))
}

# Fill with normal random values (in-place)
.tensor_methods$normal_ <- function(self, mean = 0, std = 1) {
  .Call(C_torch_normal_, self, as.double(mean), as.double(std))
}

# Fill with uniform random values (in-place)
.tensor_methods$uniform_ <- function(self, from = 0, to = 1) {
  .Call(C_torch_uniform_, self, as.double(from), as.double(to))
}

# Fill with zeros (in-place)
.tensor_methods$zero_ <- function(self) {
  .Call(C_torch_zero_, self)
}

# Fill with a scalar (in-place)
.tensor_methods$fill_ <- function(self, value) {
  .Call(C_torch_fill_, self, as.double(value))
}

# Standard deviation
.tensor_methods$std <- function(self, dim = NULL, keepdim = FALSE, unbiased = TRUE) {
  .Call(C_torch_std, self, dim, as.logical(keepdim), as.logical(unbiased))
}

# Repeat (tile) tensor
.tensor_methods$repeat_interleave <- function(self, repeats, dim = NULL) {
  .Call(C_torch_repeat_interleave, self, as.integer(repeats), dim)
}

# repeat (numpy-style tile)
.tensor_methods[["repeat"]] <- function(self, sizes) {
  .Call(C_torch_repeat, self, as.integer(sizes))
}

# Convert to R list
.tensor_methods$tolist <- function(self) {
  as.list(as.numeric(.Call(C_as_array, self)))
}

# Indexing support
.tensor_methods$index_select <- function(self, dim, index) {
  .Call(C_torch_index_select, self, as.integer(dim), index)
}

# Narrow (slice along dimension)
.tensor_methods$narrow <- function(self, dim, start, length) {
  .Call(C_torch_narrow, self, as.integer(dim), as.integer(start), as.integer(length))
}

# Scatter
.tensor_methods$scatter_ <- function(self, dim, index, src) {
  .Call(C_torch_scatter_, self, as.integer(dim), index, src)
}

# Data property (no-op, returns self since no autograd)
.tensor_properties$data <- function(self) self

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
  dev <- .Call(C_tensor_device, self)
  structure(list(type = dev), class = "torch_device")
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

#' @examples
#' \donttest{
#' x <- torch_randn(c(2, 3))
#' print(x)
#' }
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
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_add_scalar, e2, e1)  # scalar + tensor = tensor + scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_add, e1, e2, 1L)
  } else {
    .Call(C_torch_add_scalar, e1, e2)
  }
}

#' @export
`!.torch_tensor` <- function(x) {
  .Call(C_torch_logical_not, x)
}

#' @export
length.nn_buffer <- function(x) {
  .Call(C_tensor_numel, x)
}

#' @export
length.nn_parameter <- function(x) {
  .Call(C_tensor_numel, x)
}

#' @export
length.torch_tensor <- function(x) {
  .Call(C_tensor_numel, x)
}

#' @export
`-.torch_tensor` <- function(e1, e2) {
  if (missing(e2)) return(.Call(C_torch_neg, e1))
  if (!inherits(e1, "torch_tensor")) {
    # scalar - tensor = -(tensor - scalar)
    .Call(C_torch_neg, .Call(C_torch_sub_scalar, e2, e1))
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_sub, e1, e2, 1L)
  } else {
    .Call(C_torch_sub_scalar, e1, e2)
  }
}

#' @export
`*.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_mul_scalar, e2, e1)  # scalar * tensor = tensor * scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_mul, e1, e2)
  } else {
    .Call(C_torch_mul_scalar, e1, e2)
  }
}

#' @export
`/.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    # scalar / tensor: convert scalar to tensor
    e1 <- torch_tensor(e1, dtype = e2$dtype)
    .Call(C_torch_div, e1, e2)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_div, e1, e2)
  } else {
    .Call(C_torch_div_scalar, e1, e2)
  }
}

#' @export
`^.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    # scalar ^ tensor: convert scalar to tensor
    e1 <- torch_tensor(e1, dtype = e2$dtype)
    .Call(C_torch_pow, e1, e2)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_pow, e1, e2)
  } else {
    .Call(C_torch_pow_scalar, e1, e2)
  }
}

#' @export
`%%.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    e1 <- torch_tensor(e1, dtype = e2$dtype)
    .Call(C_torch_remainder, e1, e2)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_remainder, e1, e2)
  } else {
    .Call(C_torch_remainder_scalar, e1, e2)
  }
}

#' @export
`%/%.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    e1 <- torch_tensor(e1, dtype = e2$dtype)
    .Call(C_torch_floor_divide, e1, e2)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_floor_divide, e1, e2)
  } else {
    .Call(C_torch_floor_divide_scalar, e1, e2)
  }
}

#' @export
`==.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_eq_scalar, e2, e1)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_eq, e1, e2)
  } else {
    .Call(C_torch_eq_scalar, e1, e2)
  }
}

#' @export
`!=.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_ne_scalar, e2, e1)
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_ne, e1, e2)
  } else {
    .Call(C_torch_ne_scalar, e1, e2)
  }
}

#' @export
`<.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_gt_scalar, e2, e1)  # scalar < tensor = tensor > scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_lt, e1, e2)
  } else {
    .Call(C_torch_lt_scalar, e1, e2)
  }
}

#' @export
`<=.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_ge_scalar, e2, e1)  # scalar <= tensor = tensor >= scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_le, e1, e2)
  } else {
    .Call(C_torch_le_scalar, e1, e2)
  }
}

#' @export
`>.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_lt_scalar, e2, e1)  # scalar > tensor = tensor < scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_gt, e1, e2)
  } else {
    .Call(C_torch_gt_scalar, e1, e2)
  }
}

#' @export
`>=.torch_tensor` <- function(e1, e2) {
  if (!inherits(e1, "torch_tensor")) {
    .Call(C_torch_le_scalar, e2, e1)  # scalar >= tensor = tensor <= scalar
  } else if (inherits(e2, "torch_tensor")) {
    .Call(C_torch_ge, e1, e2)
  } else {
    .Call(C_torch_ge_scalar, e1, e2)
  }
}

# ---- [ indexing ----

#' @export
`[.torch_tensor` <- function(x, ..., drop = TRUE) {
  cl <- match.call(expand.dots = FALSE)
  args <- cl$...

  indices <- vector("list", length(args))
  for (i in seq_along(args)) {
    if (identical(args[[i]], quote(expr = ))) {
      # Missing argument = select all along this dimension
      indices[[i]] <- NULL
    } else {
      indices[[i]] <- eval(args[[i]], parent.frame())
    }
  }

  .Call(C_torch_index, x, indices, as.logical(drop))
}

#' @export
`[<-.torch_tensor` <- function(x, ..., value) {
  cl <- match.call(expand.dots = FALSE)
  args <- cl$...

  indices <- vector("list", length(args))
  for (i in seq_along(args)) {
    if (identical(args[[i]], quote(expr = ))) {
      indices[[i]] <- NULL
    } else {
      indices[[i]] <- eval(args[[i]], parent.frame())
    }
  }

  .Call(C_torch_index_put, x, indices, value)
  x
}
