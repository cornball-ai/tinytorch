% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_renorm_}
\alias{torch_renorm_}
\title{Renorm}
\usage{
torch_renorm_(self, p, dim, maxnorm)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A numeric scalar.}

\item{dim}{An integer.}

\item{maxnorm}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::renorm_}.
}
