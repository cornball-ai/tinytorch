% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scatter}
\alias{torch_scatter}
\title{Scatter}
\usage{
torch_scatter(self, dim, index, src)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{src}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::scatter}.
}
