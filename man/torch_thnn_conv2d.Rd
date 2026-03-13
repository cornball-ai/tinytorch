% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_thnn_conv2d}
\alias{torch_thnn_conv2d}
\title{Thnn conv2d}
\usage{
torch_thnn_conv2d(self, weight, kernel_size, bias = NULL, stride = 1,
                  padding = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[2]).}

\item{padding}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::thnn_conv2d}.
}
