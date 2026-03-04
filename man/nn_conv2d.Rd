% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_conv2d}
\alias{nn_conv2d}
\title{2D convolution module}
\usage{
nn_conv2d(in_channels, out_channels, kernel_size, stride = 1L, padding = 0L,
          dilation = 1L, groups = 1L, bias = TRUE)
}
\arguments{
\item{in_channels}{Integer.}

\item{out_channels}{Integer.}

\item{kernel_size}{Integer or vector of 2.}

\item{stride}{Integer or vector of 2. Default 1.}

\item{padding}{Integer or vector of 2. Default 0.}

\item{dilation}{Integer or vector of 2. Default 1.}

\item{groups}{Integer. Default 1.}

\item{bias}{Logical. Default TRUE.}
}
\value{
An nn_module instance.
}
\description{
2D convolution module
}
