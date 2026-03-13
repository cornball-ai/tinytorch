% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantized_max_pool1d}
\alias{torch_quantized_max_pool1d}
\title{Quantized max pool1d}
\usage{
torch_quantized_max_pool1d(self, kernel_size, stride = NULL, padding = 0,
                           dilation = 1, ceil_mode = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[1]).}

\item{stride}{(int[1]).}

\item{padding}{(int[1]).}

\item{dilation}{(int[1]).}

\item{ceil_mode}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantized_max_pool1d}.
}
