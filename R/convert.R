#' Convert a tensor to an R array
#' @param x A torch_tensor.
#' @return An R numeric vector, matrix, or array.
#' @examples
#' \donttest{
#' x <- torch_tensor(c(1, 2, 3))
#' as_array(x)
#' }
#' @export
as_array <- function(x) {
  C_as_array(x)
}

#' @examples
#' \donttest{
#' x <- torch_tensor(matrix(1:6, 2, 3))
#' as.array(x)
#' }
#' @export
as.array.torch_tensor <- function(x, ...) {
  C_as_array(x)
}

#' @examples
#' \donttest{
#' x <- torch_tensor(c(1, 2, 3))
#' as.numeric(x)
#' }
#' @export
as.numeric.torch_tensor <- function(x, ...) {
  as.numeric(C_as_array(x))
}

#' @examples
#' \donttest{
#' x <- torch_tensor(c(1, 2, 3))
#' as.double(x)
#' }
#' @export
as.double.torch_tensor <- function(x, ...) {
  as.double(C_as_array(x))
}

#' @examples
#' \donttest{
#' x <- torch_tensor(c(1, 2, 3))
#' as.integer(x)
#' }
#' @export
as.integer.torch_tensor <- function(x, ...) {
  as.integer(C_as_array(x))
}

#' @examples
#' \donttest{
#' x <- torch_tensor(matrix(1:6, 2, 3))
#' as.matrix(x)
#' }
#' @export
as.matrix.torch_tensor <- function(x, ...) {
  arr <- C_as_array(x)
  if (is.null(dim(arr))) {
    matrix(arr, nrow = length(arr), ncol = 1L)
  } else {
    as.matrix(arr)
  }
}
