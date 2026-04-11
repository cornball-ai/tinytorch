% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_squeeze}
\alias{torch_squeeze}
\title{torch_squeeze}
\usage{
torch_squeeze(self, dim = NULL)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}
}
\description{
torch_squeeze
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_zeros(c(2, 1, 2, 1, 2))
x
y = torch_squeeze(x)
y
y = torch_squeeze(x, 1)
y
y = torch_squeeze(x, 2)
y
}
}
}
