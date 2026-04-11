% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_flip}
\alias{torch_flip}
\title{Flip tensor along given dimensions}
\usage{
torch_flip(self, dims)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dims}{Integer vector of dimensions to flip.}
}
\value{
A torch_tensor.
}
\description{
Flip tensor along given dimensions
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_arange(1, 8)$view(c(2, 2, 2))
x
torch_flip(x, c(1, 2))
}
}
}
