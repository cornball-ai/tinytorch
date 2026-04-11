% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_grid_sampler_3d}
\alias{torch_grid_sampler_3d}
\title{Grid sampler 3d}
\usage{
torch_grid_sampler_3d(input, grid, interpolation_mode, padding_mode,
                      align_corners)
}
\arguments{
\item{input}{A torch_tensor.}

\item{grid}{A torch_tensor.}

\item{interpolation_mode}{An integer.}

\item{padding_mode}{An integer.}

\item{align_corners}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::grid_sampler_3d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
