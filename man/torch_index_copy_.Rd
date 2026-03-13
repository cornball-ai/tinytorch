% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_copy_}
\alias{torch_index_copy_}
\title{Index copy}
\usage{
torch_index_copy_(self, dim, index, source)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{source}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::index_copy_}.
}
