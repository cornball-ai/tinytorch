% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nextafter_}
\alias{torch_nextafter_}
\title{Nextafter}
\usage{
torch_nextafter_(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nextafter_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
