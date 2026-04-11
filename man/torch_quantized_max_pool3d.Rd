% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantized_max_pool3d}
\alias{torch_quantized_max_pool3d}
\title{Quantized max pool3d}
\usage{
torch_quantized_max_pool3d(self, kernel_size, stride = NULL, padding = 0,
                           dilation = 1, ceil_mode = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[3]).}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}

\item{dilation}{(int[3]).}

\item{ceil_mode}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantized_max_pool3d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
