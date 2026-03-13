% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_soft_margin_loss}
\alias{torch_soft_margin_loss}
\title{Soft margin loss}
\usage{
torch_soft_margin_loss(self, target, reduction = Mean)
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
Wrapper for \code{at::soft_margin_loss}.
}
