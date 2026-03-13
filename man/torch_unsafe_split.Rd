% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unsafe_split}
\alias{torch_unsafe_split}
\title{Unsafe split}
\usage{
torch_unsafe_split(self, split_size, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{split_size}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unsafe_split}.
}
