% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cdist}
\alias{torch_cdist}
\title{Cdist}
\usage{
torch_cdist(x1, x2, p = 2, compute_mode = NULL)
}
\arguments{
\item{x1}{A torch_tensor.}

\item{x2}{A torch_tensor.}

\item{p}{A numeric value.}

\item{compute_mode}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cdist}.
}
