% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hinge_embedding_loss}
\alias{torch_hinge_embedding_loss}
\title{Hinge embedding loss}
\usage{
torch_hinge_embedding_loss(self, target, margin = 1, reduction = Mean)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{margin}{A numeric value.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hinge_embedding_loss}.
}
