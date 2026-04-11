% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_copy_sparse_to_sparse_}
\alias{torch_copy_sparse_to_sparse_}
\title{Copy sparse to sparse}
\usage{
torch_copy_sparse_to_sparse_(self, src, non_blocking = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{src}{A torch_tensor.}

\item{non_blocking}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::copy_sparse_to_sparse_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
