% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_dtype_constants}
\alias{torch_dtype_constants}
\alias{torch_uint8}
\alias{torch_int8}
\alias{torch_int16}
\alias{torch_short}
\alias{torch_int}
\alias{torch_int32}
\alias{torch_int64}
\alias{torch_long}
\alias{torch_float16}
\alias{torch_half}
\alias{torch_float}
\alias{torch_float32}
\alias{torch_float64}
\alias{torch_double}
\alias{torch_cfloat}
\alias{torch_cfloat32}
\alias{torch_cdouble}
\alias{torch_cfloat64}
\alias{torch_cfloat128}
\alias{torch_bool}
\alias{torch_qint8}
\alias{torch_quint8}
\alias{torch_qint32}
\alias{torch_bfloat16}
\alias{torch_float8_e5m2}
\alias{torch_float8_e4m3fn}
\alias{torch_float8_e5m2fnuz}
\alias{torch_float8_e4m3fnuz}
\alias{torch_reduction_none}
\alias{torch_reduction_mean}
\alias{torch_reduction_sum}
\alias{torch_contiguous_format}
\alias{torch_channels_last_format}
\alias{torch_preserve_format}
\alias{torch_strided}
\alias{torch_sparse_coo}
\alias{torch_per_tensor_affine}
\alias{torch_per_channel_affine}
\alias{torch_per_tensor_symmetric}
\alias{torch_per_channel_symmetric}
\title{Torch Data Types}
\description{
Factory functions that return tensor element type objects matching
libtorch's \code{c10::ScalarType} enum. These can be called with no
arguments (e.g. \code{torch_float32()}) or passed without parens
(e.g. \code{dtype = torch_float32}); tensor creation functions accept
either form.
}
\examples{
torch_float32()
torch_int64()
}
