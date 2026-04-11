% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_renorm}
\alias{torch_renorm}
\title{Renorm}
\usage{
torch_renorm(self, p, dim, maxnorm)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A numeric scalar.}

\item{dim}{An integer.}

\item{maxnorm}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::renorm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
x = torch_ones(c(3, 3))
x[2,]$fill_(2)
x[3,]$fill_(3)
x
torch_renorm(x, 1, 1, 5)
}
}
}
