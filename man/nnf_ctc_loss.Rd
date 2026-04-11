% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_ctc_loss}
\alias{nnf_ctc_loss}
\title{Nnf ctc loss}
\usage{
nnf_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank,
             reduction, zero_infinity)
}
\arguments{
\item{log_probs}{Parameter passed to the underlying ATen operator.}

\item{targets}{Parameter passed to the underlying ATen operator.}

\item{input_lengths}{Parameter passed to the underlying ATen operator.}

\item{target_lengths}{Parameter passed to the underlying ATen operator.}

\item{blank}{Parameter passed to the underlying ATen operator.}

\item{reduction}{Parameter passed to the underlying ATen operator.}

\item{zero_infinity}{Parameter passed to the underlying ATen operator.}
}
\description{
Nnf ctc loss
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
