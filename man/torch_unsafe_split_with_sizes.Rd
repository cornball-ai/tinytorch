% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unsafe_split_with_sizes}
\alias{torch_unsafe_split_with_sizes}
\title{Unsafe split with sizes}
\usage{
torch_unsafe_split_with_sizes(self, split_sizes, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{split_sizes}{An integer vector.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unsafe_split_with_sizes}.
}
