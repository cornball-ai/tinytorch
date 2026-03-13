% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_softplus}
\alias{torch_softplus}
\title{Softplus}
\usage{
torch_softplus(self, beta = 1, threshold = 20)
}
\arguments{
\item{self}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{threshold}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::softplus}.
}
