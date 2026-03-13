% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_im2col}
\alias{torch_im2col}
\title{Im2col}
\usage{
torch_im2col(self, kernel_size, dilation, padding, stride)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{dilation}{(int[2]).}

\item{padding}{(int[2]).}

\item{stride}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::im2col}.
}
