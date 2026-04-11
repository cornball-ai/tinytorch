% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_bsr_tensor}
\alias{torch_sparse_bsr_tensor}
\title{Sparse bsr tensor}
\usage{
torch_sparse_bsr_tensor(crow_indices, col_indices, values, size, dtype = NULL,
                        device = NULL)
}
\arguments{
\item{crow_indices}{A torch_tensor.}

\item{col_indices}{A torch_tensor.}

\item{values}{A torch_tensor.}

\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sparse_bsr_tensor}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
