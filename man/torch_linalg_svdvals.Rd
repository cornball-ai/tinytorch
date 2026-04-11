% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_svdvals}
\alias{torch_linalg_svdvals}
\alias{linalg_svdvals}
\title{Linalg svdvals}
\usage{
torch_linalg_svdvals(A, driver = NULL)
}
\arguments{
\item{A}{A torch_tensor.}

\item{driver}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_svdvals}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
