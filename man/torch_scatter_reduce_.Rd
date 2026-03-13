% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scatter_reduce_}
\alias{torch_scatter_reduce_}
\title{Scatter reduce}
\usage{
torch_scatter_reduce_(self, dim, index, src, reduce, include_self = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{reduce}{A character string.}

\item{include_self}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::scatter_reduce_}.
}
