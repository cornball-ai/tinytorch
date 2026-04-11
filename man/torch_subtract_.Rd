% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_subtract_}
\alias{torch_subtract_}
\title{Subtract}
\usage{
torch_subtract_(self, other, alpha = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{alpha}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::subtract_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
