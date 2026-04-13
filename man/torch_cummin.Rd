% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cummin}
\alias{torch_cummin}
\title{Cummin}
\usage{
torch_cummin(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cummin}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
