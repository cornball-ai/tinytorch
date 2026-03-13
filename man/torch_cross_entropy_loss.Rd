% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cross_entropy_loss}
\alias{torch_cross_entropy_loss}
\title{Cross entropy loss}
\usage{
torch_cross_entropy_loss(self, target, weight = NULL, reduction = Mean,
                         ignore_index = -100, label_smoothing = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{weight}{A torch_tensor. Optional.}

\item{reduction}{An integer.}

\item{ignore_index}{An integer.}

\item{label_smoothing}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cross_entropy_loss}.
}
