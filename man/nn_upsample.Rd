% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_upsample}
\alias{nn_upsample}
\title{Upsample module}
\usage{
nn_upsample(size = NULL, scale_factor = NULL, mode = "nearest")
}
\arguments{
\item{size}{Target size (integer or integer vector).}

\item{scale_factor}{Scale factor (double or double vector).}

\item{mode}{Upsampling mode. Default "nearest".}
}
\value{
An nn_module instance.
}
\description{
Upsample module
}
\examples{
\donttest{
if (torch_is_installed()) {
input <- torch_arange(start = 1, end = 4, dtype = torch_float())$view(c(1, 1, 2, 2))
nn_upsample(scale_factor = c(2), mode = "nearest")(input)
nn_upsample(scale_factor = c(2, 2), mode = "nearest")(input)
}
}
}
