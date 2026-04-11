% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sparse_coo_tensor}
\alias{torch_sparse_coo_tensor}
\title{Sparse coo tensor}
\usage{
torch_sparse_coo_tensor(size, dtype = NULL, device = NULL)
}
\arguments{
\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sparse_coo_tensor}.
}
\examples{
\donttest{
if (torch_is_installed()) {

i = torch_tensor(matrix(c(1, 2, 2, 3, 1, 3), ncol = 3, byrow = TRUE), dtype=torch_int64())
v = torch_tensor(c(3, 4, 5), dtype=torch_float32())
torch_sparse_coo_tensor(i, v)
torch_sparse_coo_tensor(i, v, c(2, 4))

# create empty sparse tensors
S = torch_sparse_coo_tensor(
  torch_empty(c(1, 0), dtype = torch_int64()), 
  torch_tensor(numeric(), dtype = torch_float32()), 
  c(1)
)
S = torch_sparse_coo_tensor(
  torch_empty(c(1, 0), dtype = torch_int64()), 
  torch_empty(c(0, 2)), 
  c(1, 2)
)
}
}
}
