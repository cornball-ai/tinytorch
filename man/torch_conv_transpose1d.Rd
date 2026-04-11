% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv_transpose1d}
\alias{torch_conv_transpose1d}
\title{torch_conv_transpose1d}
\usage{
torch_conv_transpose1d(input, weight, bias = NULL, stride = 1L, padding = 0L,
                       output_padding = 0L, groups = 1L, dilation = 1L)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{output_padding}{Integer vector output padding.}

\item{groups}{Integer number of groups.}

\item{dilation}{Integer vector dilation.}
}
\description{
torch_conv_transpose1d
}
\examples{
\donttest{
if (torch_is_installed()) {

inputs = torch_randn(c(20, 16, 50))
weights = torch_randn(c(16, 33, 5))
nnf_conv_transpose1d(inputs, weights)
}
}
}
