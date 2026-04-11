% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_float_power_}
\alias{torch_float_power_}
\title{Float power}
\usage{
torch_float_power_(self, exponent)
}
\arguments{
\item{self}{A torch_tensor.}

\item{exponent}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::float_power_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
