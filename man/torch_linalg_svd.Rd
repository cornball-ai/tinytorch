% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_svd}
\alias{torch_linalg_svd}
\title{Linalg svd}
\usage{
torch_linalg_svd(A, full_matrices = TRUE, driver = NULL)
}
\arguments{
\item{A}{A torch_tensor.}

\item{full_matrices}{A logical value.}

\item{driver}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_svd}.
}
