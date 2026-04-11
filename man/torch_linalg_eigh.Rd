% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_eigh}
\alias{torch_linalg_eigh}
\alias{linalg_eigh}
\title{Linalg eigh}
\usage{
torch_linalg_eigh(self, UPLO = "L")
}
\arguments{
\item{self}{A torch_tensor.}

\item{UPLO}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_eigh}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
