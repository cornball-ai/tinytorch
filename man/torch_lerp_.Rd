% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lerp_}
\alias{torch_lerp_}
\title{Lerp}
\usage{
torch_lerp_(self, end, weight)
}
\arguments{
\item{self}{A torch_tensor.}

\item{end}{A torch_tensor.}

\item{weight}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lerp_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
