% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv2d}
\alias{torch_conv2d}
\title{torch_conv2d}
\usage{
torch_conv2d(input, weight, bias = NULL, stride = 1L, padding = 0L,
             dilation = 1L, groups = 1L)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{dilation}{Integer vector dilation.}

\item{groups}{Integer number of groups.}
}
\description{
torch_conv2d
}
\examples{
\donttest{
if (torch_is_installed()) {

# With square kernels and equal stride
filters = torch_randn(c(8,4,3,3))
inputs = torch_randn(c(1,4,5,5))
nnf_conv2d(inputs, filters, padding=1)
}
}
}
