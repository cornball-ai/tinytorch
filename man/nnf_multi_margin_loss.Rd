% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_multi_margin_loss}
\alias{nnf_multi_margin_loss}
\title{Nnf multi margin loss}
\usage{
nnf_multi_margin_loss(self, target, p, margin, weight, reduction)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{p}{Parameter passed to the underlying ATen operator.}

\item{margin}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\description{
Nnf multi margin loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
