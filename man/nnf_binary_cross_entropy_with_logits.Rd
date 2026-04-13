% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_binary_cross_entropy_with_logits}
\alias{nnf_binary_cross_entropy_with_logits}
\title{Nnf binary cross entropy with logits}
\usage{
nnf_binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{pos_weight}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf binary cross entropy with logits
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
