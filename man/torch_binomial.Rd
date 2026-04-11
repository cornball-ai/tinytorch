% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_binomial}
\alias{torch_binomial}
\title{Binomial}
\usage{
torch_binomial(count, prob, generator = NULL)
}
\arguments{
\item{count}{A torch_tensor.}

\item{prob}{A torch_tensor.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::binomial}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
