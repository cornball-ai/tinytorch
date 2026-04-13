% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_multilabel_margin_loss}
\alias{nnf_multilabel_margin_loss}
\title{Nnf multilabel margin loss}
\usage{
nnf_multilabel_margin_loss(self, target, reduction)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf multilabel margin loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
