% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hardshrink}
\alias{torch_hardshrink}
\title{Hardshrink}
\usage{
torch_hardshrink(self, lambd = 0.5)
}
\arguments{
\item{self}{A torch_tensor.}

\item{lambd}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hardshrink}.
}
