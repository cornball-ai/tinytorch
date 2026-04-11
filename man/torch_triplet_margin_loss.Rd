% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_triplet_margin_loss}
\alias{torch_triplet_margin_loss}
\title{Triplet margin loss}
\usage{
torch_triplet_margin_loss(anchor, positive, negative, margin = 1, p = 2,
                          eps = 1e-06, swap = FALSE, reduction = 1L)
}
\arguments{
\item{anchor}{A torch_tensor.}

\item{positive}{A torch_tensor.}

\item{negative}{A torch_tensor.}

\item{margin}{A numeric value.}

\item{p}{A numeric value.}

\item{eps}{A numeric value.}

\item{swap}{A logical value.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::triplet_margin_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
