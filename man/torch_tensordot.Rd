% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tensordot}
\alias{torch_tensordot}
\title{Tensordot}
\usage{
torch_tensordot(self, other, dims_self, dims_other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{dims_self}{An integer vector.}

\item{dims_other}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tensordot}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
