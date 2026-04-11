% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_conv3d}
\alias{nn_conv3d}
\title{Nn conv3d}
\usage{
nn_conv3d(in_channels, out_channels, kernel_size, stride = 1L, padding = 0L,
          dilation = 1L, groups = 1L, bias = TRUE)
}
\arguments{
\item{in_channels}{Parameter passed to the underlying ATen operator.}

\item{out_channels}{Parameter passed to the underlying ATen operator.}

\item{kernel_size}{Integer vector kernel size.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{dilation}{Integer vector dilation.}

\item{groups}{Integer number of groups.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}
}
\description{
Nn conv3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
