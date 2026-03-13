% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pairwise_distance}
\alias{torch_pairwise_distance}
\title{Pairwise distance}
\usage{
torch_pairwise_distance(x1, x2, p = 2, eps = 1e-06, keepdim = FALSE)
}
\arguments{
\item{x1}{A torch_tensor.}

\item{x2}{A torch_tensor.}

\item{p}{A numeric value.}

\item{eps}{A numeric value.}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pairwise_distance}.
}
