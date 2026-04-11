% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_lu_factor}
\alias{torch_linalg_lu_factor}
\alias{linalg_lu_factor}
\title{Linalg lu factor}
\usage{
torch_linalg_lu_factor(A, pivot = TRUE)
}
\arguments{
\item{A}{A torch_tensor.}

\item{pivot}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_lu_factor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
