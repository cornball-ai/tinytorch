% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_to_sparse_bsr}
\alias{torch_to_sparse_bsr}
\title{To sparse bsr}
\usage{
torch_to_sparse_bsr(self, blocksize, dense_dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{blocksize}{(int[2]).}

\item{dense_dim}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::to_sparse_bsr}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
