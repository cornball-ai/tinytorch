% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_matrix_rank}
\alias{torch_linalg_matrix_rank}
\alias{linalg_matrix_rank}
\title{Linalg matrix rank}
\usage{
torch_linalg_matrix_rank(input, atol = NULL, rtol = NULL, hermitian = FALSE)
}
\arguments{
\item{input}{A torch_tensor.}

\item{atol}{A torch_tensor. Optional.}

\item{rtol}{A torch_tensor. Optional.}

\item{hermitian}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_matrix_rank}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
