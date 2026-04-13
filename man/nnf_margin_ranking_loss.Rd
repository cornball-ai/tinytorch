% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_margin_ranking_loss}
\alias{nnf_margin_ranking_loss}
\title{Nnf margin ranking loss}
\usage{
nnf_margin_ranking_loss(input1, input2, target, margin, reduction)
}
\arguments{
\item{input1}{Parameter passed to the underlying ATen operator.}

\item{input2}{Parameter passed to the underlying ATen operator.}

\item{target}{Parameter passed to the underlying ATen operator.}

\item{margin}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf margin ranking loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
