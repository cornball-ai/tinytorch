% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_pinv}
\alias{torch_linalg_pinv}
\title{Linalg pinv}
\usage{
torch_linalg_pinv(self, atol = NULL, rtol = NULL, hermitian = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{atol}{A torch_tensor. Optional.}

\item{rtol}{A torch_tensor. Optional.}

\item{hermitian}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_pinv}.
}
