% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_max_pool3d}
\alias{nnf_max_pool3d}
\title{Nnf max pool3d}
\usage{
nnf_max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{kernel_size}{Integer vector kernel size.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{dilation}{Integer vector dilation.}

\item{ceil_mode}{Parameter passed to the underlying ATen operator.}
}
\description{
Nnf max pool3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
