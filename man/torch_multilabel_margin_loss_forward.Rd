% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_multilabel_margin_loss_forward}
\alias{torch_multilabel_margin_loss_forward}
\title{Multilabel margin loss forward}
\usage{
torch_multilabel_margin_loss_forward(self, target, reduction)
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
Wrapper for \code{at::multilabel_margin_loss_forward}.
}
