% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_transpose}
\alias{torch_transpose}
\title{torch_transpose}
\usage{
torch_transpose(self, dim0, dim1)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim0}{Integer dimension.}

\item{dim1}{Integer dimension.}
}
\description{
torch_transpose
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(2, 3))
x
torch_transpose(x, 1, 2)
}
}
}
