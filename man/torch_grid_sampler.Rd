% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_grid_sampler}
\alias{torch_grid_sampler}
\title{Grid sampler}
\usage{
torch_grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners)
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
Wrapper for \code{at::grid_sampler}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
