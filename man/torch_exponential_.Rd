% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_exponential_}
\alias{torch_exponential_}
\title{Exponential}
\usage{
torch_exponential_(self, lambd = 1, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{lambd}{A numeric value.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::exponential_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
