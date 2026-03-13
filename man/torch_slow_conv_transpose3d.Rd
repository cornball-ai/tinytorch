% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_slow_conv_transpose3d}
\alias{torch_slow_conv_transpose3d}
\title{Slow conv transpose3d}
\usage{
torch_slow_conv_transpose3d(self, weight, kernel_size, bias = NULL, stride = 1,
                            padding = 0, output_padding = 0, dilation = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{kernel_size}{(int[3]).}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}

\item{output_padding}{(int[3]).}

\item{dilation}{(int[3]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::slow_conv_transpose3d}.
}
