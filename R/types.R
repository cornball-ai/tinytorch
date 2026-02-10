# Dtype constants matching c10::ScalarType enum values
# These integer codes are passed to C++ and cast to at::ScalarType

#' @export
torch_float16 <- structure(5L, class = "torch_dtype")

#' @export
torch_float <- structure(6L, class = "torch_dtype")

#' @export
torch_float32 <- torch_float

#' @export
torch_float64 <- structure(7L, class = "torch_dtype")

#' @export
torch_double <- torch_float64

#' @export
torch_int8 <- structure(1L, class = "torch_dtype")

#' @export
torch_int16 <- structure(2L, class = "torch_dtype")

#' @export
torch_int <- structure(3L, class = "torch_dtype")

#' @export
torch_int32 <- torch_int

#' @export
torch_int64 <- structure(4L, class = "torch_dtype")

#' @export
torch_long <- torch_int64

#' @export
torch_bool <- structure(11L, class = "torch_dtype")

#' Print a torch_dtype
#' @param x A torch_dtype.
#' @param ... Ignored.
#' @export
print.torch_dtype <- function(x, ...) {
  names <- c(
    "torch_uint8", "torch_int8", "torch_int16", "torch_int32",
    "torch_int64", "torch_float16", "torch_float32", "torch_float64",
    "torch_complex32", "torch_complex64", "torch_complex128", "torch_bool"
  )
  idx <- unclass(x) + 1L
  cat(names[idx], "\n")
  invisible(x)
}

#' @export
as.character.torch_dtype <- function(x, ...) {
  # Match torch R package convention
  names <- c(
    "Byte", "Char", "Short", "Int", "Long",
    "Half", "Float", "Double",
    "ComplexHalf", "ComplexFloat", "ComplexDouble", "Bool"
  )
  idx <- unclass(x) + 1L
  if (idx >= 1L && idx <= length(names)) names[idx] else sprintf("Unknown(%d)", unclass(x))
}

# Map scalar type codes to human-readable names
.dtype_name <- function(code) {
  names <- c(
    "UInt8", "Int8", "Int16", "Int32", "Int64",
    "Float16", "Float32", "Float64",
    "ComplexHalf", "ComplexFloat", "ComplexDouble", "Bool"
  )
  idx <- code + 1L
  if (idx >= 1L && idx <= length(names)) names[idx] else sprintf("Unknown(%d)", code)
}
