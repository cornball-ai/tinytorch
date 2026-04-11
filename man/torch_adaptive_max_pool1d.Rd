% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_adaptive_max_pool1d}
\alias{torch_adaptive_max_pool1d}
\title{Adaptive max pool1d}
\usage{
torch_adaptive_max_pool1d(self, output_size)
}
\arguments{
\item{self}{A torch_tensor.}

\item{output_size}{(int[1]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::adaptive_max_pool1d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
