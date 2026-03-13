% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_to_sparse_csc}
\alias{torch_to_sparse_csc}
\title{To sparse csc}
\usage{
torch_to_sparse_csc(self, dense_dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dense_dim}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::to_sparse_csc}.
}
