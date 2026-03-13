% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_addr_}
\alias{torch_addr_}
\title{Addr}
\usage{
torch_addr_(self, vec1, vec2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{vec1}{A torch_tensor.}

\item{vec2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::addr_}.
}
