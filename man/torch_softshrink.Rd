% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_softshrink}
\alias{torch_softshrink}
\title{Softshrink}
\usage{
torch_softshrink(self, lambd = 0.5)
}
\arguments{
\item{self}{A torch_tensor.}

\item{lambd}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::softshrink}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
