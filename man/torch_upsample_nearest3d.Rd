% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_upsample_nearest3d}
\alias{torch_upsample_nearest3d}
\title{Upsample nearest3d}
\usage{
torch_upsample_nearest3d(input, output_size, scale_factors)
}
\arguments{
\item{input}{A torch_tensor.}

\item{output_size}{An integer vector.}

\item{scale_factors}{A numeric vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::upsample_nearest3d}.
}
