% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_upsample_bicubic2d}
\alias{torch_upsample_bicubic2d}
\title{Upsample bicubic2d}
\usage{
torch_upsample_bicubic2d(input, output_size, align_corners, scale_factors)
}
\arguments{
\item{input}{A torch_tensor.}

\item{output_size}{An integer vector.}

\item{align_corners}{A logical value.}

\item{scale_factors}{A numeric vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::upsample_bicubic2d}.
}
