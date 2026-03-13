% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ormqr}
\alias{torch_ormqr}
\title{Ormqr}
\usage{
torch_ormqr(self, input2, input3, left = TRUE, transpose = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{input2}{A torch_tensor.}

\item{input3}{A torch_tensor.}

\item{left}{A logical value.}

\item{transpose}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::ormqr}.
}
