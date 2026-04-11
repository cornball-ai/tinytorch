% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_native_norm}
\alias{torch_native_norm}
\title{Native norm}
\usage{
torch_native_norm(self, p = 2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::native_norm}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
