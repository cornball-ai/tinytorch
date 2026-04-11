% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_tensorinv}
\alias{torch_linalg_tensorinv}
\alias{linalg_tensorinv}
\title{Linalg tensorinv}
\usage{
torch_linalg_tensorinv(self, ind = 2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{ind}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_tensorinv}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
