% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_multi_margin_loss}
\alias{nn_multi_margin_loss}
\title{Nn multi margin loss}
\usage{
nn_multi_margin_loss(p = 1, margin = 1, weight = NULL, reduction = 1L)
}
\arguments{
\item{p}{Parameter passed to the underlying ATen operator.}

\item{margin}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\description{
Nn multi margin loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
