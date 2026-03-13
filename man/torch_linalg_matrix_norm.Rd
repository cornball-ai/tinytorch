% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_matrix_norm}
\alias{torch_linalg_matrix_norm}
\title{Linalg matrix norm}
\usage{
torch_linalg_matrix_norm(self, ord, dim = NULL, keepdim = FALSE, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{ord}{A numeric scalar.}

\item{dim}{An integer vector.}

\item{keepdim}{A logical value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_matrix_norm}.
}
