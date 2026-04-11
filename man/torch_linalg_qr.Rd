% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_qr}
\alias{torch_linalg_qr}
\alias{linalg_qr}
\title{Linalg qr}
\usage{
torch_linalg_qr(A, mode = "reduced")
}
\arguments{
\item{A}{A torch_tensor.}

\item{mode}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_qr}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
