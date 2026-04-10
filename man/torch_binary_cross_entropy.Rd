% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_binary_cross_entropy}
\alias{torch_binary_cross_entropy}
\title{Binary cross entropy}
\usage{
torch_binary_cross_entropy(self, target, weight = NULL, reduction = 1L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{weight}{A torch_tensor. Optional.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::binary_cross_entropy}.
}
