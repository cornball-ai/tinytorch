% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_clamp_max_}
\alias{torch_clamp_max_}
\title{Clamp max}
\usage{
torch_clamp_max_(self, max)
}
\arguments{
\item{self}{A torch_tensor.}

\item{max}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::clamp_max_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
