% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_take_along_dim}
\alias{torch_take_along_dim}
\title{Take along dim}
\usage{
torch_take_along_dim(self, indices, dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{indices}{A torch_tensor.}

\item{dim}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::take_along_dim}.
}
\examples{
\donttest{
if (torch_is_installed()) {
t <- torch_tensor(matrix(c(10, 30, 20, 60, 40, 50), nrow = 2))
max_idx <- torch_argmax(t)
torch_take_along_dim(t, max_idx)

sorted_idx <- torch_argsort(t, dim=2)
torch_take_along_dim(t, sorted_idx, dim=2)

}
}
}
