% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_clamp_min_}
\alias{torch_clamp_min_}
\title{Clamp min}
\usage{
torch_clamp_min_(self, min)
}
\arguments{
\item{self}{A torch_tensor.}

\item{min}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::clamp_min_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
