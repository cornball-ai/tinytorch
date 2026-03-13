% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_norm_except_dim}
\alias{torch_norm_except_dim}
\title{Norm except dim}
\usage{
torch_norm_except_dim(v, pow = 2, dim = 0)
}
\arguments{
\item{v}{A torch_tensor.}

\item{pow}{An integer.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::norm_except_dim}.
}
