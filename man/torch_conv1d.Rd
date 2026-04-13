% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv1d}
\alias{torch_conv1d}
\title{1D convolution}
\usage{
torch_conv1d(input, weight, bias = NULL, stride = 1L, padding = 0L,
             dilation = 1L, groups = 1L)
}
\arguments{
\item{input}{Input tensor of shape (N, C_in, L).}

\item{weight}{Filter tensor of shape (C_out, C_in/groups, kW).}

\item{bias}{Optional bias tensor of shape (C_out).}

\item{stride}{Stride of the convolution. Default 1.}

\item{padding}{Zero-padding added to both sides. Default 0.}

\item{dilation}{Spacing between kernel elements. Default 1.}

\item{groups}{Number of blocked connections. Default 1.}
}
\value{
A `torch_tensor`.
}
\description{
1D convolution
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(1, 1, 10))
w <- torch_randn(c(1, 1, 3))
torch_conv1d(x, w)
}
}
}
