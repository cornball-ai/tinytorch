% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_narrow}
\alias{torch_narrow}
\title{torch_narrow}
\usage{
torch_narrow(self, dim, start, length)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}

\item{start}{Integer start index.}

\item{length}{Integer length.}
}
\description{
torch_narrow
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_tensor(matrix(c(1:9), ncol = 3, byrow= TRUE))
torch_narrow(x, 1, 1, 2)
torch_narrow(x, 2, 2, 2)
}
}
}
