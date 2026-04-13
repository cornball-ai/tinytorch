% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_flatten}
\alias{torch_flatten}
\title{torch_flatten}
\usage{
torch_flatten(self, start_dim = 1L, end_dim = -1L)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{start_dim}{Integer starting dimension.}

\item{end_dim}{Integer ending dimension.}
}
\value{
A `torch_tensor`.
}
\description{
torch_flatten
}
\examples{
\donttest{
if (torch_is_installed()) {

t = torch_tensor(matrix(c(1, 2), ncol = 2))
torch_flatten(t)
torch_flatten(t, start_dim=2)
}
}
}
