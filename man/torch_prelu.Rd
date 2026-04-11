% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_prelu}
\alias{torch_prelu}
\title{Prelu}
\usage{
torch_prelu(self, weight)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::prelu}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
