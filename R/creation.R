#' Create a tensor from R data
#'
#' @param data An R numeric, integer, or logical vector/matrix/array.
#' @param dtype Optional torch dtype (e.g. torch_float).
#' @param device Ignored for now (CPU only).
#' @return A torch_tensor object.
#' @export
torch_tensor <- function(data, dtype = NULL, device = NULL) {
  if (inherits(data, "torch_tensor")) return(data)
  dtype_code <- if (!is.null(dtype)) unclass(dtype) else NULL
  .Call(C_torch_tensor, data, dtype_code, device)
}

#' Create a tensor of zeros
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @export
torch_zeros <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- if (!is.null(dtype)) unclass(dtype) else NULL
  .Call(C_torch_zeros, size, dtype_code)
}

#' Create a tensor of ones
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @export
torch_ones <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- if (!is.null(dtype)) unclass(dtype) else NULL
  .Call(C_torch_ones, size, dtype_code)
}

#' Create a tensor with random normal values
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @export
torch_randn <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- if (!is.null(dtype)) unclass(dtype) else NULL
  .Call(C_torch_randn, size, dtype_code)
}

#' Create an uninitialized tensor with same shape/dtype as input
#' @param self A torch_tensor.
#' @export
torch_empty_like <- function(self) {
  .Call(C_torch_empty_like, self)
}

#' Create an uninitialized tensor
#' @param ... Dimensions (as separate integers or a single integer vector).
#' @param dtype Optional torch dtype.
#' @param device Ignored (CPU only).
#' @export
torch_empty <- function(..., dtype = NULL, device = NULL) {
  size <- as.integer(c(...))
  dtype_code <- if (!is.null(dtype)) unclass(dtype) else NULL
  .Call(C_torch_empty, size, dtype_code)
}

#' Check if two tensors are element-wise close
#'
#' @param input A torch_tensor.
#' @param other A torch_tensor.
#' @param rtol Relative tolerance (default 1e-05).
#' @param atol Absolute tolerance (default 1e-08).
#' @return Logical scalar.
#' @export
torch_allclose <- function(input, other, rtol = 1e-05, atol = 1e-08) {
  diff <- (input - other)$abs()
  max_diff <- diff$max()$item()
  max_other <- other$abs()$max()$item()
  max_diff <= atol + rtol * max_other
}
