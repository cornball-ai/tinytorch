% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_cross_entropy_loss}
\alias{nn_cross_entropy_loss}
\title{Nn cross entropy loss}
\usage{
nn_cross_entropy_loss(weight = NULL, reduction = 1L, ignore_index = -100,
                      label_smoothing = 0)
}
\arguments{
\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{ignore_index}{Parameter passed to the underlying ATen operator.}

\item{label_smoothing}{Parameter passed to the underlying ATen operator.}
}
\description{
Nn cross entropy loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
