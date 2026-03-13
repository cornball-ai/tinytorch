% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_resize_and_clear_}
\alias{torch_sparse_resize_and_clear_}
\title{Sparse resize and clear}
\usage{
torch_sparse_resize_and_clear_(self, size, sparse_dim, dense_dim)
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
Wrapper for \code{at::sparse_resize_and_clear_}.
}
