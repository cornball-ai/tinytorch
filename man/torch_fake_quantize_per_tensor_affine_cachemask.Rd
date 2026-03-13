% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fake_quantize_per_tensor_affine_cachemask}
\alias{torch_fake_quantize_per_tensor_affine_cachemask}
\title{Fake quantize per tensor affine cachemask}
\usage{
torch_fake_quantize_per_tensor_affine_cachemask(self, scale, zero_point,
                                                quant_min, quant_max)
}
\arguments{
\item{self}{A torch_tensor.}

\item{scale}{A numeric value.}

\item{zero_point}{An integer.}

\item{quant_min}{An integer.}

\item{quant_max}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fake_quantize_per_tensor_affine_cachemask}.
}
