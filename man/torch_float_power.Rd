% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_float_power}
\alias{torch_float_power}
\title{Float power}
\usage{
torch_float_power(self, exponent)
}
\arguments{
\item{self}{A torch_tensor.}

\item{exponent}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::float_power}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
