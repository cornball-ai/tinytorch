% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pin_memory}
\alias{torch_pin_memory}
\title{Pin memory}
\usage{
torch_pin_memory(self, device = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pin_memory}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
