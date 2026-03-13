% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_fill_}
\alias{torch_index_fill_}
\title{Index fill}
\usage{
torch_index_fill_(self, dim, index, value)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{value}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::index_fill_}.
}
