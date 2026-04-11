% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diff}
\alias{torch_diff}
\title{Diff}
\usage{
torch_diff(self, n = 1, dim = -1, prepend = NULL, append = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{n}{An integer.}

\item{dim}{An integer.}

\item{prepend}{A torch_tensor. Optional.}

\item{append}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diff}.
}
\examples{
\donttest{
if (torch_is_installed()) {
a <- torch_tensor(c(1,2,3))
torch_diff(a)

b <- torch_tensor(c(4, 5))
torch_diff(a, append = b)

c <- torch_tensor(rbind(c(1,2,3), c(3,4,5)))
torch_diff(c, dim = 1)
torch_diff(c, dim = 2) 

}
}
}
