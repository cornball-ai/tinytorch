% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pow_}
\alias{torch_pow_}
\title{Pow}
\usage{
torch_pow_(self, exponent)
}
\arguments{
\item{self}{A torch_tensor.}

\item{exponent}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pow_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
