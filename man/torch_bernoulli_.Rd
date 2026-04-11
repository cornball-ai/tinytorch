% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bernoulli_}
\alias{torch_bernoulli_}
\title{Bernoulli}
\usage{
torch_bernoulli_(self, p, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A torch_tensor.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bernoulli_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
