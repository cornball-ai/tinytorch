#' Convert a tensor to an R array
#' @param x A torch_tensor.
#' @return An R numeric vector, matrix, or array.
#' @export
as_array <- function(x) {
  .Call(C_as_array, x)
}

#' @export
as.array.torch_tensor <- function(x, ...) {
  .Call(C_as_array, x)
}

#' @export
as.numeric.torch_tensor <- function(x, ...) {
  as.numeric(.Call(C_as_array, x))
}

#' @export
as.double.torch_tensor <- function(x, ...) {
  as.double(.Call(C_as_array, x))
}

#' @export
as.integer.torch_tensor <- function(x, ...) {
  as.integer(.Call(C_as_array, x))
}

#' @export
as.matrix.torch_tensor <- function(x, ...) {
  arr <- .Call(C_as_array, x)
  if (is.null(dim(arr))) {
    matrix(arr, nrow = length(arr), ncol = 1L)
  } else {
    as.matrix(arr)
  }
}
