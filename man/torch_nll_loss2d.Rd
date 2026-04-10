% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nll_loss2d}
\alias{torch_nll_loss2d}
\title{Nll loss2d}
\usage{
torch_nll_loss2d(self, target, weight = NULL, reduction = 1L,
                 ignore_index = -100)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{weight}{A torch_tensor. Optional.}

\item{reduction}{An integer.}

\item{ignore_index}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nll_loss2d}.
}
