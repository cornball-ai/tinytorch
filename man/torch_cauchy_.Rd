% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cauchy_}
\alias{torch_cauchy_}
\title{Cauchy}
\usage{
torch_cauchy_(self, median = 0, sigma = 1, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{median}{A numeric value.}

\item{sigma}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cauchy_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
