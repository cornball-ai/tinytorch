% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_binary_cross_entropy}
\alias{nnf_binary_cross_entropy}
\title{Nnf binary cross entropy}
\usage{
nnf_binary_cross_entropy(self, target, weight, reduction)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\description{
Nnf binary cross entropy
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
