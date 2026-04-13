% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_triplet_margin_loss}
\alias{nnf_triplet_margin_loss}
\title{Nnf triplet margin loss}
\usage{
nnf_triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap,
                        reduction)
}
\arguments{
\item{anchor}{Parameter passed to the underlying ATen operator.}

\item{positive}{Parameter passed to the underlying ATen operator.}

\item{negative}{Parameter passed to the underlying ATen operator.}

\item{margin}{Parameter passed to the underlying ATen operator.}

\item{p}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{swap}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf triplet margin loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
