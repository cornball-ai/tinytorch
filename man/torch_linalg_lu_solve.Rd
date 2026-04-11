% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_lu_solve}
\alias{torch_linalg_lu_solve}
\alias{linalg_lu_solve}
\title{Linalg lu solve}
\usage{
torch_linalg_lu_solve(LU, pivots, B, left = TRUE, adjoint = FALSE)
}
\arguments{
\item{LU}{A torch_tensor.}

\item{pivots}{A torch_tensor.}

\item{B}{A torch_tensor.}

\item{left}{A logical value.}

\item{adjoint}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_lu_solve}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
