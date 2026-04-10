% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mse_loss}
\alias{torch_mse_loss}
\title{Mse loss}
\usage{
torch_mse_loss(self, target, reduction = 1L)
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
Wrapper for \code{at::mse_loss}.
}
