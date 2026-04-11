% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_celu}
\alias{torch_celu}
\title{Celu}
\usage{
torch_celu(self, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::celu}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
