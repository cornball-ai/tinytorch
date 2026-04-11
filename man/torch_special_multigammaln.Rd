% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_special_multigammaln}
\alias{torch_special_multigammaln}
\title{Special multigammaln}
\usage{
torch_special_multigammaln(self, p)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::special_multigammaln}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
