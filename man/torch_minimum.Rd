% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_minimum}
\alias{torch_minimum}
\title{Minimum}
\usage{
torch_minimum(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::minimum}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, -1))
b <- torch_tensor(c(3, 0, 4))
torch_minimum(a, b)
}
}
}
