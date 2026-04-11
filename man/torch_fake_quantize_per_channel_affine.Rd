% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fake_quantize_per_channel_affine}
\alias{torch_fake_quantize_per_channel_affine}
\title{Fake quantize per channel affine}
\usage{
torch_fake_quantize_per_channel_affine(self, scale, zero_point, axis,
                                       quant_min, quant_max)
}
\arguments{
\item{self}{A torch_tensor.}

\item{scale}{A torch_tensor.}

\item{zero_point}{A torch_tensor.}

\item{axis}{An integer.}

\item{quant_min}{An integer.}

\item{quant_max}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fake_quantize_per_channel_affine}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
