% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pixel_unshuffle}
\alias{torch_pixel_unshuffle}
\title{Pixel unshuffle}
\usage{
torch_pixel_unshuffle(self, downscale_factor)
}
\arguments{
\item{self}{A torch_tensor.}

\item{downscale_factor}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pixel_unshuffle}.
}
