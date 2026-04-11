% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_eigvalsh}
\alias{torch_linalg_eigvalsh}
\alias{linalg_eigvalsh}
\title{Linalg eigvalsh}
\usage{
torch_linalg_eigvalsh(self, UPLO = "L")
}
\arguments{
\item{self}{A torch_tensor.}

\item{UPLO}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_eigvalsh}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
