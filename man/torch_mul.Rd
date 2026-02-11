% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mul}
\alias{torch_mul}
\title{Multiply two tensors element-wise}
\usage{
torch_mul(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor or scalar.}
}
\description{
Multiply two tensors element-wise
}
\examples{
\donttest{
a <- torch_randn(c(2, 3))
torch_mul(a, 2.0)
}
}
