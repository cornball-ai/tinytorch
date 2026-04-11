% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rot90}
\alias{torch_rot90}
\title{Rot90}
\usage{
torch_rot90(self, k = 1, dims = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{k}{An integer.}

\item{dims}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rot90}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_arange(1, 4)$view(c(2, 2))
x
torch_rot90(x, 1, c(1, 2))
x <- torch_arange(1, 8)$view(c(2, 2, 2))
x
torch_rot90(x, 1, c(1, 2))
}
}
}
