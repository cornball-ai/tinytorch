% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_conv3d}
\alias{torch_conv3d}
\title{Conv3d}
\usage{
torch_conv3d(input, weight, bias = NULL, stride = 1, padding = 0, dilation = 1,
             groups = 1)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor. Optional.}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}

\item{dilation}{(int[3]).}

\item{groups}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::conv3d}.
}
\examples{
\donttest{
if (torch_is_installed()) {

# filters = torch_randn(c(33, 16, 3, 3, 3))
# inputs = torch_randn(c(20, 16, 50, 10, 20))
# nnf_conv3d(inputs, filters)
}
}
}
