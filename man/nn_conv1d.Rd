% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_conv1d}
\alias{nn_conv1d}
\title{1D convolution module}
\usage{
nn_conv1d(in_channels, out_channels, kernel_size, stride = 1L, padding = 0L,
          dilation = 1L, groups = 1L, bias = TRUE)
}
\arguments{
\item{in_channels}{Integer.}

\item{out_channels}{Integer.}

\item{kernel_size}{Integer.}

\item{stride}{Integer. Default 1.}

\item{padding}{Integer. Default 0.}

\item{dilation}{Integer. Default 1.}

\item{groups}{Integer. Default 1.}

\item{bias}{Logical. Default TRUE.}
}
\value{
An nn_module instance.
}
\description{
1D convolution module
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- nn_conv1d(16, 33, 3, stride = 2)
input <- torch_randn(20, 16, 50)
output <- m(input)
}
}
}
