% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv_transpose3d}
\alias{torch_conv_transpose3d}
\title{Conv transpose3d}
\usage{
torch_conv_transpose3d(input, weight, bias = NULL, stride = 1, padding = 0,
                       output_padding = 0, groups = 1, dilation = 1)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}

\item{output_padding}{(int[3]).}

\item{groups}{An integer.}

\item{dilation}{(int[3]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::conv_transpose3d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
