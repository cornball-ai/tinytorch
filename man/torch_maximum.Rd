% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_maximum}
\alias{torch_maximum}
\title{Element-wise maximum of two tensors}
\usage{
torch_maximum(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Element-wise maximum of two tensors
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(1, 2, -1))
b <- torch_tensor(c(3, 0, 4))
torch_maximum(a, b)
}
}
}
