% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hsplit}
\alias{torch_hsplit}
\title{Hsplit}
\usage{
torch_hsplit(self, sections)
}
\arguments{
\item{self}{A torch_tensor.}

\item{sections}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hsplit}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
