% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_as_strided_scatter}
\alias{torch_as_strided_scatter}
\title{As strided scatter}
\usage{
torch_as_strided_scatter(self, src, size, stride, storage_offset = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{size}{An integer vector.}

\item{stride}{An integer vector.}

\item{storage_offset}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::as_strided_scatter}.
}
