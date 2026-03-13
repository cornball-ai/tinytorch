% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_norm}
\alias{torch_linalg_norm}
\title{Linalg norm}
\usage{
torch_linalg_norm(self, ord = NULL, dim = NULL, keepdim = FALSE, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{ord}{A numeric scalar. Optional.}

\item{dim}{(int[1]). Optional.}

\item{keepdim}{A logical value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_norm}.
}
