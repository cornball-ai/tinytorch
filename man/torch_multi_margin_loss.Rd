% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_multi_margin_loss}
\alias{torch_multi_margin_loss}
\title{Multi margin loss}
\usage{
torch_multi_margin_loss(self, target, p = 1, margin = 1, weight = NULL,
                        reduction = 1L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{target}{A torch_tensor.}

\item{p}{A numeric scalar.}

\item{margin}{A numeric scalar.}

\item{weight}{A torch_tensor. Optional.}

\item{reduction}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::multi_margin_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
