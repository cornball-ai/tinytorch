% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logcumsumexp}
\alias{torch_logcumsumexp}
\title{Logcumsumexp}
\usage{
torch_logcumsumexp(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logcumsumexp}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_randn(c(10))
torch_logcumsumexp(a, dim=1)
}
}
}
