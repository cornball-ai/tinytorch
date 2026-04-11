% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_ldl_solve}
\alias{torch_linalg_ldl_solve}
\alias{linalg_ldl_solve}
\title{Linalg ldl solve}
\usage{
torch_linalg_ldl_solve(LD, pivots, B, hermitian = FALSE)
}
\arguments{
\item{LD}{A torch_tensor.}

\item{pivots}{A torch_tensor.}

\item{B}{A torch_tensor.}

\item{hermitian}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_ldl_solve}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
