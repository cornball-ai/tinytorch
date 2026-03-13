% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv_depthwise3d}
\alias{torch_conv_depthwise3d}
\title{Conv depthwise3d}
\usage{
torch_conv_depthwise3d(self, weight, kernel_size, bias, stride, padding,
                       dilation)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{kernel_size}{(int[3]).}

\item{bias}{A torch_tensor.}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}

\item{dilation}{(int[3]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::conv_depthwise3d}.
}
