% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_linear}
\alias{nnf_linear}
\title{Nnf linear}
\usage{
nnf_linear(input, weight, bias)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf linear
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
