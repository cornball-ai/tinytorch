% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sum}
\alias{torch_sum}
\title{Sum of tensor elements}
\usage{
torch_sum(self, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Optional dimension to reduce (1-based).}

\item{keepdim}{Whether to keep the reduced dimension.}
}
\description{
Sum of tensor elements
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
torch_sum(x)
torch_sum(x, dim = 1)
}
}
}
