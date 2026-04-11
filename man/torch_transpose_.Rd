% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_transpose_}
\alias{torch_transpose_}
\title{Transpose}
\usage{
torch_transpose_(self, dim0, dim1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim0}{An integer.}

\item{dim1}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::transpose_}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
