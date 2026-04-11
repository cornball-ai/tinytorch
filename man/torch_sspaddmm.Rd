% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sspaddmm}
\alias{torch_sspaddmm}
\title{Sspaddmm}
\usage{
torch_sspaddmm(self, mat1, mat2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mat1}{A torch_tensor.}

\item{mat2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sspaddmm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
