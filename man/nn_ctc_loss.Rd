% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_ctc_loss}
\alias{nn_ctc_loss}
\title{Nn ctc loss}
\usage{
nn_ctc_loss(blank = 0L, reduction = 1L, zero_infinity = FALSE)
}
\arguments{
\item{blank}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{zero_infinity}{Parameter passed to the underlying ATen operator.}
}
\description{
Nn ctc loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
