% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_conv2d}
\alias{nnf_conv2d}
\title{Nnf conv2d}
\usage{
nnf_conv2d(input, weight, bias, stride, padding, dilation, groups)
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
\value{
A `torch_tensor`.
}
\description{
Nnf conv2d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
