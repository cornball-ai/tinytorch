% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bincount}
\alias{torch_bincount}
\title{Bincount}
\usage{
torch_bincount(self, weights = NULL, minlength = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weights}{A torch_tensor. Optional.}

\item{minlength}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bincount}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
