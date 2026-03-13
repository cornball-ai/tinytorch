% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_frobenius_norm}
\alias{torch_frobenius_norm}
\title{Frobenius norm}
\usage{
torch_frobenius_norm(self, dim, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]).}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::frobenius_norm}.
}
