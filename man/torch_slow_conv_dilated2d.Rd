% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_slow_conv_dilated2d}
\alias{torch_slow_conv_dilated2d}
\title{Slow conv dilated2d}
\usage{
torch_slow_conv_dilated2d(self, weight, kernel_size, bias = NULL, stride = 1,
                          padding = 0, dilation = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[2]).}

\item{padding}{(int[2]).}

\item{dilation}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::slow_conv_dilated2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
