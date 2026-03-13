% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_slice_scatter}
\alias{torch_slice_scatter}
\title{Slice scatter}
\usage{
torch_slice_scatter(self, src, dim = 0, start = NULL, end = NULL, step = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{dim}{An integer.}

\item{start}{An integer. Optional.}

\item{end}{An integer. Optional.}

\item{step}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::slice_scatter}.
}
