% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_upsample_bilinear2d}
\alias{torch_upsample_bilinear2d}
\title{Upsample bilinear2d}
\usage{
torch_upsample_bilinear2d(input, output_size, align_corners, scale_factors)
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
Wrapper for \code{at::upsample_bilinear2d}.
}
