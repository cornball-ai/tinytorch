% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_cholesky}
\alias{torch_linalg_cholesky}
\alias{linalg_cholesky}
\title{Linalg cholesky}
\usage{
torch_linalg_cholesky(self, upper = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{upper}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_cholesky}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
