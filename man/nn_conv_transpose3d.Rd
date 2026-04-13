% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_conv_transpose3d}
\alias{nn_conv_transpose3d}
\title{Nn conv transpose3d}
\usage{
nn_conv_transpose3d(in_channels, out_channels, kernel_size, stride = 1L,
                    padding = 0L, output_padding = 0L, groups = 1L,
                    bias = TRUE, dilation = 1L)
}
\arguments{
\item{in_channels}{Parameter passed to the underlying ATen operator.}

\item{out_channels}{Parameter passed to the underlying ATen operator.}

\item{kernel_size}{Integer vector kernel size.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{output_padding}{Integer vector output padding.}

\item{groups}{Integer number of groups.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{dilation}{Integer vector dilation.}
}
\value{
An `nn_module` instance.
}
\description{
Nn conv transpose3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
