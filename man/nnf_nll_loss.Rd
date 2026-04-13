% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_nll_loss}
\alias{nnf_nll_loss}
\title{Nnf nll loss}
\usage{
nnf_nll_loss(self, target, weight, reduction, ignore_index)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{weight}{A `torch_tensor` of weights.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{ignore_index}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf nll loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
