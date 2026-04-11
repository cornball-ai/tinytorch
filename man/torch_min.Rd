% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_min}
\alias{torch_min}
\title{torch_min}
\usage{
torch_min(self, dim = NULL)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}
}
\description{
torch_min
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_min(a)


a = torch_randn(c(4, 4))
a
torch_min(a, dim = 1)


a = torch_randn(c(4))
a
b = torch_randn(c(4))
b
torch_min(a, other = b)
}
}
}
