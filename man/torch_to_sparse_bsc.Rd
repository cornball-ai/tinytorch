% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_to_sparse_bsc}
\alias{torch_to_sparse_bsc}
\title{To sparse bsc}
\usage{
torch_to_sparse_bsc(self, blocksize, dense_dim = NULL)
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
Wrapper for \code{at::to_sparse_bsc}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
