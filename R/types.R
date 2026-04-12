#' Torch Data Types
#'
#' Factory functions that return tensor element type objects matching
#' libtorch's \code{c10::ScalarType} enum. These can be called with no
#' arguments (e.g. \code{torch_float32()}) or passed without parens
#' (e.g. \code{dtype = torch_float32}); tensor creation functions accept
#' either form.
#'
#' @name torch_dtype_constants
#' @examples
#' torch_float32()
#' torch_int64()
NULL

.make_dtype <- function(code) {
  code <- as.integer(code)
  function() structure(code, class = "torch_dtype")
}

#' @rdname torch_dtype_constants
#' @export
torch_uint8 <- .make_dtype(0L)

#' @rdname torch_dtype_constants
#' @export
torch_int8 <- .make_dtype(1L)

#' @rdname torch_dtype_constants
#' @export
torch_int16 <- .make_dtype(2L)

#' @rdname torch_dtype_constants
#' @export
torch_short <- torch_int16

#' @rdname torch_dtype_constants
#' @export
torch_int <- .make_dtype(3L)

#' @rdname torch_dtype_constants
#' @export
torch_int32 <- torch_int

#' @rdname torch_dtype_constants
#' @export
torch_int64 <- .make_dtype(4L)

#' @rdname torch_dtype_constants
#' @export
torch_long <- torch_int64

#' @rdname torch_dtype_constants
#' @export
torch_float16 <- .make_dtype(5L)

#' @rdname torch_dtype_constants
#' @export
torch_half <- torch_float16

#' @rdname torch_dtype_constants
#' @export
torch_float <- .make_dtype(6L)

#' @rdname torch_dtype_constants
#' @export
torch_float32 <- torch_float

#' @rdname torch_dtype_constants
#' @export
torch_float64 <- .make_dtype(7L)

#' @rdname torch_dtype_constants
#' @export
torch_double <- torch_float64

#' @rdname torch_dtype_constants
#' @export
torch_cfloat <- .make_dtype(9L)

#' @rdname torch_dtype_constants
#' @export
torch_cfloat32 <- torch_cfloat

#' @rdname torch_dtype_constants
#' @export
torch_cdouble <- .make_dtype(10L)

#' @rdname torch_dtype_constants
#' @export
torch_cfloat64 <- torch_cdouble

#' @rdname torch_dtype_constants
#' @export
torch_cfloat128 <- torch_cdouble

#' @rdname torch_dtype_constants
#' @export
torch_bool <- .make_dtype(11L)

#' @rdname torch_dtype_constants
#' @export
torch_qint8 <- .make_dtype(12L)

#' @rdname torch_dtype_constants
#' @export
torch_quint8 <- .make_dtype(13L)

#' @rdname torch_dtype_constants
#' @export
torch_qint32 <- .make_dtype(14L)

#' @rdname torch_dtype_constants
#' @export
torch_bfloat16 <- .make_dtype(15L)

#' @rdname torch_dtype_constants
#' @export
torch_float8_e5m2 <- .make_dtype(23L)

#' @rdname torch_dtype_constants
#' @export
torch_float8_e4m3fn <- .make_dtype(24L)

#' @rdname torch_dtype_constants
#' @export
torch_float8_e5m2fnuz <- .make_dtype(25L)

#' @rdname torch_dtype_constants
#' @export
torch_float8_e4m3fnuz <- .make_dtype(26L)

# Resolve a dtype argument to a raw integer code, accepting:
# - NULL                 (returns NULL)
# - a torch_dtype object (returns its underlying code)
# - a dtype factory      (calls it, then unclasses)
.dtype_code <- function(d) {
  if (is.null(d)) return(NULL)
  if (is.function(d)) d <- d()
  unclass(d)
}

#' Print a torch_dtype
#' @param x A torch_dtype.
#' @param ... Ignored.
#' @examples
#' print(torch_float32())
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

#' As.character.torch dtype
#' @param x Parameter passed to the underlying ATen operator.
#' @param ... Additional arguments passed to methods.
#' @examples
#' as.character(torch_float32())
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
