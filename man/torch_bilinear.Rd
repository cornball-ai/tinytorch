% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bilinear}
\alias{torch_bilinear}
\title{Bilinear}
\usage{
torch_bilinear(input1, input2, weight, bias = NULL)
}
\arguments{
\item{input1}{A torch_tensor.}

\item{input2}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bilinear}.
}
