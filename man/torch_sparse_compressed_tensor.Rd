% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_compressed_tensor}
\alias{torch_sparse_compressed_tensor}
\title{Sparse compressed tensor}
\usage{
torch_sparse_compressed_tensor(compressed_indices, plain_indices, values, size,
                               dtype = NULL, device = NULL)
}
\arguments{
\item{compressed_indices}{A torch_tensor.}

\item{plain_indices}{A torch_tensor.}

\item{values}{A torch_tensor.}

\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sparse_compressed_tensor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
