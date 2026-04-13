% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lu}
\alias{torch_lu}
\title{torch_lu}
\usage{
torch_lu(A, pivot = TRUE)
}
\arguments{
\item{A}{A `torch_tensor` (matrix).}

\item{pivot}{Logical; enable pivoting.}
}
\value{
A `torch_tensor`.
}
\description{
torch_lu
}
\examples{
\donttest{
if (torch_is_installed()) {

A <- torch_randn(c(2, 3, 3))
torch_lu(A)
}
}
}
