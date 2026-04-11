% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cov}
\alias{torch_cov}
\title{Cov}
\usage{
torch_cov(self, correction = 1, fweights = NULL, aweights = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{correction}{An integer.}

\item{fweights}{A torch_tensor. Optional.}

\item{aweights}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cov}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
