% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_reshape}
\alias{torch_reshape}
\title{torch_reshape}
\usage{
torch_reshape(self, shape)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{shape}{Integer vector shape.}
}
\description{
torch_reshape
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_arange(0, 3)
torch_reshape(a, list(2, 2))
b <- torch_tensor(matrix(c(0, 1, 2, 3), ncol = 2, byrow=TRUE))
torch_reshape(b, list(-1))
}
}
}
