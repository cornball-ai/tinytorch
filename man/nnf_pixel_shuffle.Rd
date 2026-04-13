% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_pixel_shuffle}
\alias{nnf_pixel_shuffle}
\title{Nnf pixel shuffle}
\usage{
nnf_pixel_shuffle(self, upscale_factor)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{upscale_factor}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf pixel shuffle
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
