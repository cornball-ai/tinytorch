% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_add}
\alias{torch_index_add}
\title{Index add}
\usage{
torch_index_add(self, dim, index, source, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{source}{A torch_tensor.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::index_add}.
}
