% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_gather}
\alias{torch_gather}
\title{torch_gather}
\usage{
torch_gather(self, dim, index)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}

\item{index}{A `torch_tensor` of indices.}
}
\value{
A `torch_tensor`.
}
\description{
torch_gather
}
\examples{
\donttest{
if (torch_is_installed()) {

t = torch_tensor(matrix(c(1,2,3,4), ncol = 2, byrow = TRUE))
torch_gather(t, 2, torch_tensor(matrix(c(1,1,2,1), ncol = 2, byrow=TRUE), dtype = torch_int64()))
}
}
}
