% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_bilinear}
\alias{nnf_bilinear}
\title{Nnf bilinear}
\usage{
nnf_bilinear(input1, input2, weight, bias)
}
\arguments{
\item{input1}{Parameter passed to the underlying ATen operator.}

\item{input2}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}
}
\description{
Nnf bilinear
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
