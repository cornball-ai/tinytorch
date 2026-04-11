% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantize_per_channel}
\alias{torch_quantize_per_channel}
\title{Quantize per channel}
\usage{
torch_quantize_per_channel(self, scales, zero_points, axis, dtype)
}
\arguments{
\item{self}{A torch_tensor.}

\item{scales}{A torch_tensor.}

\item{zero_points}{A torch_tensor.}

\item{axis}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantize_per_channel}.
}
\examples{
\donttest{
if (torch_is_installed()) {
x = torch_tensor(matrix(c(-1.0, 0.0, 1.0, 2.0), ncol = 2, byrow = TRUE))
torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
                           torch_tensor(c(10L, 0L)), 0, torch_quint8())
torch_quantize_per_channel(x, torch_tensor(c(0.1, 0.01)), 
                           torch_tensor(c(10L, 0L)), 0, torch_quint8())$int_repr()
}
}
}
