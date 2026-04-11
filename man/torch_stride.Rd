% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_stride}
\alias{torch_stride}
\title{Stride}
\usage{
torch_stride(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{A dimension name.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::stride}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
