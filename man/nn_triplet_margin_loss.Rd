% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_triplet_margin_loss}
\alias{nn_triplet_margin_loss}
\title{Nn triplet margin loss}
\usage{
nn_triplet_margin_loss(margin = 1, p = 2, eps = 1e-06, swap = FALSE,
                       reduction = 1L)
}
\arguments{
\item{margin}{Parameter passed to the underlying ATen operator.}

\item{p}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{swap}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn triplet margin loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
