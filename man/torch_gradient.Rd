% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_gradient}
\alias{torch_gradient}
\title{Gradient}
\usage{
torch_gradient(self, spacing = NULL, dim = NULL, edge_order = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{spacing}{A numeric scalar. Optional.}

\item{dim}{An integer. Optional.}

\item{edge_order}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::gradient}.
}
