% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_conv_transpose1d}
\alias{nnf_conv_transpose1d}
\title{Nnf conv transpose1d}
\usage{
nnf_conv_transpose1d(input, weight, bias, stride, padding, output_padding,
                     groups, dilation)
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
Nnf conv transpose1d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
