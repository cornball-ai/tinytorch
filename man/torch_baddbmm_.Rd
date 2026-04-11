% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_baddbmm_}
\alias{torch_baddbmm_}
\title{Baddbmm}
\usage{
torch_baddbmm_(self, batch1, batch2, beta = 1, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{batch1}{A torch_tensor.}

\item{batch2}{A torch_tensor.}

\item{beta}{A numeric scalar.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::baddbmm_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
