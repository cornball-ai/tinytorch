% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_dist}
\alias{torch_dist}
\title{Dist}
\usage{
torch_dist(self, other, p = 2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{p}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::dist}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(4))
x
y = torch_randn(c(4))
y
torch_dist(x, y, 3.5)
torch_dist(x, y, 3)
torch_dist(x, y, 0)
torch_dist(x, y, 1)
}
}
}
