% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addmv}
\alias{torch_addmv}
\title{Addmv}
\usage{
torch_addmv(self, mat, vec, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mat}{A torch_tensor.}

\item{vec}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addmv}.
}
\examples{
\donttest{
if (torch_is_installed()) {

M = torch_randn(c(2))
mat = torch_randn(c(2, 3))
vec = torch_randn(c(3))
torch_addmv(M, mat, vec)
}
}
}
