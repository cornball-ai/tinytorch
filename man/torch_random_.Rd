% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_random_}
\alias{torch_random_}
\title{Random}
\usage{
torch_random_(self, from, to, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{from}{An integer.}

\item{to}{An integer.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::random_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
