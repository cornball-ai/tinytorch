% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_dsplit}
\alias{torch_dsplit}
\title{Dsplit}
\usage{
torch_dsplit(self, sections)
}
\arguments{
\item{self}{A torch_tensor.}

\item{sections}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::dsplit}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
