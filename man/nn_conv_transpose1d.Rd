% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_conv_transpose1d}
\alias{nn_conv_transpose1d}
\title{1D transposed convolution module}
\usage{
nn_conv_transpose1d(in_channels, out_channels, kernel_size, stride = 1L,
                    padding = 0L, output_padding = 0L, groups = 1L,
                    bias = TRUE, dilation = 1L)
}
\arguments{
\item{in_channels}{Integer.}

\item{out_channels}{Integer.}

\item{kernel_size}{Integer.}

\item{stride}{Integer. Default 1.}

\item{padding}{Integer. Default 0.}

\item{output_padding}{Integer. Default 0.}

\item{groups}{Integer. Default 1.}

\item{bias}{Logical. Default TRUE.}

\item{dilation}{Integer. Default 1.}
}
\value{
An nn_module instance.
}
\description{
1D transposed convolution module
}
