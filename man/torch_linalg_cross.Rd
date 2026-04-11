% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_cross}
\alias{torch_linalg_cross}
\alias{linalg_cross}
\title{Linalg cross}
\usage{
torch_linalg_cross(self, other, dim = -1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_cross}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
