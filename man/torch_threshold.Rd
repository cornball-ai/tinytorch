% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_threshold}
\alias{torch_threshold}
\title{Threshold}
\usage{
torch_threshold(self, threshold, value)
}
\arguments{
\item{self}{A torch_tensor.}

\item{threshold}{A numeric scalar.}

\item{value}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::threshold}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
