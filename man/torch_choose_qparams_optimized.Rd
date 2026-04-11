% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_choose_qparams_optimized}
\alias{torch_choose_qparams_optimized}
\title{Choose qparams optimized}
\usage{
torch_choose_qparams_optimized(input, numel, n_bins, ratio, bit_width)
}
\arguments{
\item{input}{A torch_tensor.}

\item{numel}{An integer.}

\item{n_bins}{An integer.}

\item{ratio}{A numeric value.}

\item{bit_width}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::choose_qparams_optimized}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
