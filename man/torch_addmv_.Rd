% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addmv_}
\alias{torch_addmv_}
\title{Addmv}
\usage{
torch_addmv_(self, mat, vec, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mat}{A torch_tensor.}

\item{vec}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addmv_}.
}
