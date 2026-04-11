% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_resize_}
\alias{torch_sparse_resize_}
\title{Sparse resize}
\usage{
torch_sparse_resize_(self, size, sparse_dim, dense_dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{sparse_dim}{An integer.}

\item{dense_dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sparse_resize_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
