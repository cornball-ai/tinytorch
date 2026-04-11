% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_mask}
\alias{torch_sparse_mask}
\title{Sparse mask}
\usage{
torch_sparse_mask(self, mask)
}
\arguments{
\item{self}{A torch_tensor.}

\item{mask}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sparse_mask}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
