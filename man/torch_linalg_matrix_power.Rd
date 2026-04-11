% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_matrix_power}
\alias{torch_linalg_matrix_power}
\alias{linalg_matrix_power}
\title{Linalg matrix power}
\usage{
torch_linalg_matrix_power(self, n)
}
\arguments{
\item{self}{A torch_tensor.}

\item{n}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_matrix_power}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
