% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ctc_loss}
\alias{torch_ctc_loss}
\title{Ctc loss}
\usage{
torch_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank = 0,
               reduction = 1L, zero_infinity = FALSE)
}
\arguments{
\item{log_probs}{A torch_tensor.}

\item{targets}{A torch_tensor.}

\item{input_lengths}{An integer vector.}

\item{target_lengths}{An integer vector.}

\item{blank}{An integer.}

\item{reduction}{An integer.}

\item{zero_infinity}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::ctc_loss}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
