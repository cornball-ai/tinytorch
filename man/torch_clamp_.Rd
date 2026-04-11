% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_clamp_}
\alias{torch_clamp_}
\title{Clamp}
\usage{
torch_clamp_(self, min = NULL, max = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{min}{A numeric scalar. Optional.}

\item{max}{A numeric scalar. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::clamp_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
