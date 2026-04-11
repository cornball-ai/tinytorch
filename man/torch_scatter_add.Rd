% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scatter_add}
\alias{torch_scatter_add}
\title{Scatter add}
\usage{
torch_scatter_add(self, dim, index, src)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{index}{A torch_tensor.}

\item{src}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::scatter_add}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
