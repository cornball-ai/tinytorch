% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cumsum}
\alias{torch_cumsum}
\title{Cumulative sum along a dimension}
\usage{
torch_cumsum(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to compute cumsum along.}
}
\value{
A torch_tensor.
}
\description{
Cumulative sum along a dimension
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(10))
a
torch_cumsum(a, dim=1)
}
}
}
