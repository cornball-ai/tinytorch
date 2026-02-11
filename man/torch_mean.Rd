% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mean}
\alias{torch_mean}
\title{Mean of tensor elements}
\usage{
torch_mean(self, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Optional dimension to reduce (1-based).}

\item{keepdim}{Whether to keep the reduced dimension.}
}
\description{
Mean of tensor elements
}
\examples{
\donttest{
x <- torch_randn(c(2, 3))
torch_mean(x)
torch_mean(x, dim = 1)
}
}
