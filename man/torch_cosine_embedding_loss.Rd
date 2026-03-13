% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cosine_embedding_loss}
\alias{torch_cosine_embedding_loss}
\title{Cosine embedding loss}
\usage{
torch_cosine_embedding_loss(input1, input2, target, margin = 0, reduction = Mean)
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
Wrapper for \code{at::cosine_embedding_loss}.
}
