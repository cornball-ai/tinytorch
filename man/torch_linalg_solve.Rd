% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_solve}
\alias{torch_linalg_solve}
\alias{linalg_solve}
\title{Linalg solve}
\usage{
torch_linalg_solve(A, B, left = TRUE)
}
\arguments{
\item{A}{A torch_tensor.}

\item{B}{A torch_tensor.}

\item{left}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_solve}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
