% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_quantize_per_channel}
\alias{torch_quantize_per_channel}
\title{Quantize per channel}
\usage{
torch_quantize_per_channel(self, scales, zero_points, axis, dtype)
}
\arguments{
\item{self}{A torch_tensor.}

\item{scales}{A torch_tensor.}

\item{zero_points}{A torch_tensor.}

\item{axis}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::quantize_per_channel}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
