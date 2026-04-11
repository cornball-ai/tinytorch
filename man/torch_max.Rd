% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_max}
\alias{torch_max}
\title{torch_max}
\usage{
torch_max(self, dim = NULL)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}
}
\description{
torch_max
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_max(a)


a = torch_randn(c(4, 4))
a
torch_max(a, dim = 1)


a = torch_randn(c(4))
a
b = torch_randn(c(4))
b
torch_max(a, other = b)
}
}
}
