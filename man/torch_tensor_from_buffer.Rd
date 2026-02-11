% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tensor_from_buffer}
\alias{torch_tensor_from_buffer}
\title{Create a tensor from raw bytes buffer}
\usage{
torch_tensor_from_buffer(raw, shape, dtype, device = NULL)
}
\arguments{
\item{raw}{Raw vector of bytes.}

\item{shape}{Integer vector of dimensions.}

\item{dtype}{A torch_dtype.}
}
\value{
A torch_tensor.
}
\description{
Create a tensor from raw bytes buffer
}
