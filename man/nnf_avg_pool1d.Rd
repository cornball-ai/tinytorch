% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_avg_pool1d}
\alias{nnf_avg_pool1d}
\title{1D average pooling}
\usage{
nnf_avg_pool1d(input, kernel_size, stride = kernel_size, padding = 0L,
               ceil_mode = FALSE, count_include_pad = TRUE)
}
\arguments{
\item{input}{A torch_tensor of shape (N, C, L).}

\item{kernel_size}{Integer, size of the pooling window.}

\item{stride}{Integer, stride of the pooling window. Default kernel_size.}

\item{padding}{Integer, zero-padding on both sides. Default 0.}

\item{ceil_mode}{Logical. Use ceil instead of floor for output size. Default FALSE.}

\item{count_include_pad}{Logical. Include padding in average. Default TRUE.}
}
\value{
A torch_tensor.
}
\description{
Applies 1D average pooling over an input signal.
}
