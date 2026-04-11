% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantize_per_tensor}
\alias{torch_quantize_per_tensor}
\title{Quantize per tensor}
\usage{
torch_quantize_per_tensor(self, scale, zero_point, dtype)
}
\arguments{
\item{self}{A torch_tensor.}

\item{scale}{A numeric value.}

\item{zero_point}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantize_per_tensor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())
torch_quantize_per_tensor(torch_tensor(c(-1.0, 0.0, 1.0, 2.0)), 0.1, 10, torch_quint8())$int_repr()
}
}
}
