% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cummax}
\alias{torch_cummax}
\title{Cummax}
\usage{
torch_cummax(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cummax}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
