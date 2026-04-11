% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unfold}
\alias{torch_unfold}
\title{Unfold}
\usage{
torch_unfold(self, dimension, size, step)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dimension}{An integer.}

\item{size}{An integer.}

\item{step}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unfold}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
