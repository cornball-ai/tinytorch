% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sub}
\alias{torch_sub}
\title{Subtract two tensors}
\usage{
torch_sub(self, other, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{alpha}{Scalar multiplier for other (default 1).}
}
\value{
A `torch_tensor`.
}
\description{
Subtract two tensors
}
\examples{
\donttest{
if (torch_is_installed()) {
a <- torch_ones(c(2, 3))
b <- torch_ones(c(2, 3))
torch_sub(a, b)
}
}
}
