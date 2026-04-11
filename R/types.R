#' Torch Data Types
#'
#' Constants representing tensor element types, matching libtorch's
#' \code{c10::ScalarType} enum. Pass these to \code{dtype} arguments
#' in tensor creation functions.
#'
#' @name torch_dtype_constants
#' @examples
#' torch_float32
#' torch_int64
NULL

#' @rdname torch_dtype_constants
#' @export
torch_uint8 <- structure(0L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float16 <- structure(5L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float <- structure(6L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float32 <- torch_float

#' @rdname torch_dtype_constants
#' @export
torch_float64 <- structure(7L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_double <- torch_float64

#' @rdname torch_dtype_constants
#' @export
torch_int8 <- structure(1L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_int16 <- structure(2L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_int <- structure(3L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_int32 <- torch_int

#' @rdname torch_dtype_constants
#' @export
torch_int64 <- structure(4L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_long <- torch_int64

#' @rdname torch_dtype_constants
#' @export
torch_bool <- structure(11L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_bfloat16 <- structure(15L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float8_e5m2 <- structure(23L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float8_e4m3fn <- structure(24L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float8_e5m2fnuz <- structure(25L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_float8_e4m3fnuz <- structure(26L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_half <- torch_float16

#' @rdname torch_dtype_constants
#' @export
torch_short <- torch_int16

#' @rdname torch_dtype_constants
#' @export
torch_cfloat <- structure(9L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_cfloat32 <- torch_cfloat

#' @rdname torch_dtype_constants
#' @export
torch_cdouble <- structure(10L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_cfloat64 <- torch_cdouble

#' @rdname torch_dtype_constants
#' @export
torch_cfloat128 <- torch_cdouble

#' @rdname torch_dtype_constants
#' @export
torch_qint8 <- structure(12L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_quint8 <- structure(13L, class = "torch_dtype")

#' @rdname torch_dtype_constants
#' @export
torch_qint32 <- structure(14L, class = "torch_dtype")

#' Print a torch_dtype
#' @param x A torch_dtype.
#' @param ... Ignored.
#' @examples
#' print(torch_float32)
#' @export
print.torch_dtype <- function(x, ...) {
  names <- c(
    "torch_uint8", "torch_int8", "torch_int16", "torch_int32",
    "torch_int64", "torch_float16", "torch_float32", "torch_float64",
    "torch_complex32", "torch_complex64", "torch_complex128", "torch_bool",
    "torch_qint8", "torch_quint8", "torch_qint32", "torch_bfloat16",
    "torch_quint4x2", "torch_quint2x4",
    "torch_bits1x8", "torch_bits2x4", "torch_bits4x2", "torch_bits8", "torch_bits16",
    "torch_float8_e5m2", "torch_float8_e4m3fn",
    "torch_float8_e5m2fnuz", "torch_float8_e4m3fnuz"
  )
  idx <- unclass(x) + 1L
  cat(names[idx], "\n")
  invisible(x)
}

#' @examples
#' as.character(torch_float32)
#' @export
as.character.torch_dtype <- function(x, ...) {
  # Match torch R package convention
  names <- c(
    "Byte", "Char", "Short", "Int", "Long",
    "Half", "Float", "Double",
    "ComplexHalf", "ComplexFloat", "ComplexDouble", "Bool",
    "QInt8", "QUInt8", "QInt32", "BFloat16",
    "QUInt4x2", "QUInt2x4",
    "Bits1x8", "Bits2x4", "Bits4x2", "Bits8", "Bits16",
    "Float8_e5m2", "Float8_e4m3fn",
    "Float8_e5m2fnuz", "Float8_e4m3fnuz"
  )
  idx <- unclass(x) + 1L
  if (idx >= 1L && idx <= length(names)) names[idx] else sprintf("Unknown(%d)", unclass(x))
}

# Map scalar type codes to human-readable names
.dtype_name <- function(code) {
  names <- c(
    "UInt8", "Int8", "Int16", "Int32", "Int64",
    "Float16", "Float32", "Float64",
    "ComplexHalf", "ComplexFloat", "ComplexDouble", "Bool",
    "QInt8", "QUInt8", "QInt32", "BFloat16",
    "QUInt4x2", "QUInt2x4",
    "Bits1x8", "Bits2x4", "Bits4x2", "Bits8", "Bits16",
    "Float8_e5m2", "Float8_e4m3fn",
    "Float8_e5m2fnuz", "Float8_e4m3fnuz"
  )
  idx <- code + 1L
  if (idx >= 1L && idx <= length(names)) names[idx] else sprintf("Unknown(%d)", code)
}
