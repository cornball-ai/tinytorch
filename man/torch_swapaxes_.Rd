% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_swapaxes_}
\alias{torch_swapaxes_}
\title{Swapaxes}
\usage{
torch_swapaxes_(self, axis0, axis1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{axis0}{An integer.}

\item{axis1}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::swapaxes_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
