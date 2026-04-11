% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_linalg_lstsq}
\alias{torch_linalg_lstsq}
\alias{linalg_lstsq}
\title{Linalg lstsq}
\usage{
torch_linalg_lstsq(self, b, rcond = NULL, driver = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{b}{A torch_tensor.}

\item{rcond}{A numeric value. Optional.}

\item{driver}{A character string. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::linalg_lstsq}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
