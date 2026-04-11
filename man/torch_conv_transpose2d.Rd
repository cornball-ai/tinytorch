% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv_transpose2d}
\alias{torch_conv_transpose2d}
\title{Conv transpose2d}
\usage{
torch_conv_transpose2d(input, weight, bias = NULL, stride = 1, padding = 0,
                       output_padding = 0, groups = 1, dilation = 1)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[2]).}

\item{padding}{(int[2]).}

\item{output_padding}{(int[2]).}

\item{groups}{An integer.}

\item{dilation}{(int[2]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::conv_transpose2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {

# With square kernels and equal stride
inputs = torch_randn(c(1, 4, 5, 5))
weights = torch_randn(c(4, 8, 3, 3))
nnf_conv_transpose2d(inputs, weights, padding=1)
}
}
}
