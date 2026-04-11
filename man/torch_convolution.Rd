% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_convolution}
\alias{torch_convolution}
\title{Convolution}
\usage{
torch_convolution(input, weight, bias, stride, padding, dilation, transposed,
                  output_padding, groups)
}
\arguments{
\item{input}{A torch_tensor.}

\item{weight}{A torch_tensor.}

\item{bias}{A torch_tensor.}

\item{stride}{An integer vector.}

\item{padding}{An integer vector.}

\item{dilation}{An integer vector.}

\item{transposed}{A logical value.}

\item{output_padding}{An integer vector.}

\item{groups}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::convolution}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
