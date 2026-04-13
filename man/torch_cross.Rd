% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cross}
\alias{torch_cross}
\title{Cross}
\usage{
torch_cross(self, other, dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{dim}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cross}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
