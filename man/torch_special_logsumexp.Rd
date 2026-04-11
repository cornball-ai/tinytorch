% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_special_logsumexp}
\alias{torch_special_logsumexp}
\title{Special logsumexp}
\usage{
torch_special_logsumexp(self, dim, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]).}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::special_logsumexp}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
