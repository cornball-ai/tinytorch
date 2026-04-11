% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pdist}
\alias{torch_pdist}
\title{Pdist}
\usage{
torch_pdist(self, p = 2)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pdist}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
