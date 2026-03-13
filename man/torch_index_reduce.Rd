% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_reduce}
\alias{torch_index_reduce}
\title{Index reduce}
\usage{
torch_index_reduce(self, dim, index, source, reduce, include_self = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{source}{A torch_tensor.}

\item{reduce}{A character string.}

\item{include_self}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::index_reduce}.
}
