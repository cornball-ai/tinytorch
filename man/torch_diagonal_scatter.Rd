% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diagonal_scatter}
\alias{torch_diagonal_scatter}
\title{Diagonal scatter}
\usage{
torch_diagonal_scatter(self, src, offset = 0, dim1 = 0, dim2 = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{offset}{An integer.}

\item{dim1}{An integer.}

\item{dim2}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diagonal_scatter}.
}
