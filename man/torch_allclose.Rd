% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_allclose}
\alias{torch_allclose}
\title{Check if two tensors are element-wise close}
\usage{
torch_allclose(input, other, rtol = 1e-05, atol = 1e-08)
}
\arguments{
\item{input}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{rtol}{Relative tolerance (default 1e-05).}

\item{atol}{Absolute tolerance (default 1e-08).}
}
\value{
Logical scalar.
}
\description{
Check if two tensors are element-wise close
}
\examples{
\donttest{
a <- torch_ones(c(2, 3))
b <- torch_ones(c(2, 3))
torch_allclose(a, b)
}
}
