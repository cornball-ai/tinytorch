% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_upsample_trilinear3d}
\alias{torch_upsample_trilinear3d}
\title{Upsample trilinear3d}
\usage{
torch_upsample_trilinear3d(input, output_size, align_corners, scale_factors)
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
Wrapper for \code{at::upsample_trilinear3d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
