% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_huber_loss}
\alias{torch_huber_loss}
\title{Huber loss}
\usage{
torch_huber_loss(self, target, reduction = 1L, delta = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{reduction}{An integer.}

\item{delta}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::huber_loss}.
}
