% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantile}
\alias{torch_quantile}
\title{Quantile}
\usage{
torch_quantile(self, q, dim = NULL, keepdim = FALSE, interpolation = "linear")
}
\arguments{
\item{self}{A torch_tensor.}

\item{q}{A torch_tensor.}

\item{dim}{An integer. Optional.}

\item{keepdim}{A logical value.}

\item{interpolation}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantile}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_randn(c(1, 3))
a
q <- torch_tensor(c(0, 0.5, 1))
torch_quantile(a, q)


a <- torch_randn(c(2, 3))
a
q <- torch_tensor(c(0.25, 0.5, 0.75))
torch_quantile(a, q, dim=1, keepdim=TRUE)
torch_quantile(a, q, dim=1, keepdim=TRUE)$shape
}
}
}
