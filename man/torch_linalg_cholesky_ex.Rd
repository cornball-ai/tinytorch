% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_cholesky_ex}
\alias{torch_linalg_cholesky_ex}
\alias{linalg_cholesky_ex}
\title{Linalg cholesky ex}
\usage{
torch_linalg_cholesky_ex(self, upper = FALSE, check_errors = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{upper}{A logical value.}

\item{check_errors}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_cholesky_ex}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
