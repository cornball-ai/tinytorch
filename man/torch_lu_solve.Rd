% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lu_solve}
\alias{torch_lu_solve}
\title{Lu solve}
\usage{
torch_lu_solve(self, LU_data, LU_pivots)
}
\arguments{
\item{self}{A torch_tensor.}

\item{LU_data}{A torch_tensor.}

\item{LU_pivots}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lu_solve}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
