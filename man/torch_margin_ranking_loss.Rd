% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_margin_ranking_loss}
\alias{torch_margin_ranking_loss}
\title{Margin ranking loss}
\usage{
torch_margin_ranking_loss(input1, input2, target, margin = 0, reduction = 1L)
}
\arguments{
\item{input1}{A torch_tensor.}

\item{input2}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{margin}{A numeric value.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::margin_ranking_loss}.
}
