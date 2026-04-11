% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_vander}
\alias{torch_linalg_vander}
\alias{linalg_vander}
\title{Linalg vander}
\usage{
torch_linalg_vander(x, N = NULL)
}
\arguments{
\item{x}{A torch_tensor.}

\item{N}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_vander}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
