% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tile}
\alias{torch_tile}
\title{Tile}
\usage{
torch_tile(self, dims)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dims}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tile}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
