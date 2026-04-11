% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_slow_conv3d_forward}
\alias{torch_slow_conv3d_forward}
\title{Slow conv3d forward}
\usage{
torch_slow_conv3d_forward(self, weight, kernel_size, bias, stride, padding)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{kernel_size}{(int[3]).}

\item{bias}{A torch_tensor.}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::slow_conv3d_forward}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
