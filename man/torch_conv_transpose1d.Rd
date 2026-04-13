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
\value{
A `torch_tensor`.
}
\description{
torch_conv_transpose1d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
