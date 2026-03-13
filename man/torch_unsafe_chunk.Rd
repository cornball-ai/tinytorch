% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unsafe_chunk}
\alias{torch_unsafe_chunk}
\title{Unsafe chunk}
\usage{
torch_unsafe_chunk(self, chunks, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{chunks}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unsafe_chunk}.
}
