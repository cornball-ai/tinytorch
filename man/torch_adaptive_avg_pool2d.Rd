% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_adaptive_avg_pool2d}
\alias{torch_adaptive_avg_pool2d}
\title{Adaptive avg pool2d}
\usage{
torch_adaptive_avg_pool2d(self, output_size)
}
\arguments{
\item{self}{A torch_tensor.}

\item{output_size}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::adaptive_avg_pool2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
