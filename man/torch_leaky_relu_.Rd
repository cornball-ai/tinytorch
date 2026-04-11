% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_leaky_relu_}
\alias{torch_leaky_relu_}
\title{Leaky relu}
\usage{
torch_leaky_relu_(self, negative_slope = 0.01)
}
\arguments{
\item{self}{A torch_tensor.}

\item{negative_slope}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::leaky_relu_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
