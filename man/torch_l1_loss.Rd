% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_l1_loss}
\alias{torch_l1_loss}
\title{L1 loss}
\usage{
torch_l1_loss(self, target, reduction = Mean)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::l1_loss}.
}
