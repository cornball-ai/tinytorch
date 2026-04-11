% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_interpolate}
\alias{nnf_interpolate}
\title{Interpolate a tensor}
\usage{
nnf_interpolate(input, size = NULL, scale_factor = NULL, mode = "nearest",
                align_corners = NULL, recompute_scale_factor = NULL)
}
\arguments{
\item{input}{A torch_tensor.}

\item{size}{Integer vector of output spatial sizes.}

\item{scale_factor}{Numeric multiplier for spatial size.}

\item{mode}{Interpolation mode: "nearest", "linear", "bilinear", "bicubic",
"trilinear", or "area". Default "nearest".}

\item{align_corners}{Logical. If TRUE, aligns corner pixels. Default NULL.}

\item{recompute_scale_factor}{Logical. Ignored (for API compatibility).}
}
\value{
A torch_tensor.
}
\description{
Down/up samples the input to the given size or scale_factor.
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(1, 1, 4, 4))
nnf_interpolate(x, size = c(8L, 8L), mode = "nearest")
}
}
}
