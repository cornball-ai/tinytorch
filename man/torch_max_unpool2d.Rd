% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_max_unpool2d}
\alias{torch_max_unpool2d}
\title{Max unpool2d}
\usage{
torch_max_unpool2d(self, indices, output_size)
}
\arguments{
\item{self}{A torch_tensor.}

\item{indices}{A torch_tensor.}

\item{output_size}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::max_unpool2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
