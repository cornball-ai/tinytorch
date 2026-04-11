% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_remainder}
\alias{torch_remainder}
\title{torch_remainder}
\usage{
torch_remainder(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\description{
torch_remainder
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_remainder(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
torch_remainder(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
}
}
}
