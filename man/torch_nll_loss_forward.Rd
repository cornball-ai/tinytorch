% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nll_loss_forward}
\alias{torch_nll_loss_forward}
\title{Nll loss forward}
\usage{
torch_nll_loss_forward(self, target, weight, reduction, ignore_index)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{reduction}{An integer.}

\item{ignore_index}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nll_loss_forward}.
}
