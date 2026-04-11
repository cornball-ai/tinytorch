% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addr}
\alias{torch_addr}
\title{Addr}
\usage{
torch_addr(self, vec1, vec2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{vec1}{A torch_tensor.}

\item{vec2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addr}.
}
\examples{
\donttest{
if (torch_is_installed()) {

vec1 = torch_arange(1, 3)
vec2 = torch_arange(1, 2)
M = torch_zeros(c(3, 2))
torch_addr(M, vec1, vec2)
}
}
}
