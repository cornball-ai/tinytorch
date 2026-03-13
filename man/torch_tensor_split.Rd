% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tensor_split}
\alias{torch_tensor_split}
\title{Tensor split}
\usage{
torch_tensor_split(self, sections, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{sections}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tensor_split}.
}
