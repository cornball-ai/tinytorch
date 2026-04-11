% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_affine_grid_generator}
\alias{torch_affine_grid_generator}
\title{Affine grid generator}
\usage{
torch_affine_grid_generator(theta, size, align_corners)
}
\arguments{
\item{theta}{A torch_tensor.}

\item{size}{An integer vector.}

\item{align_corners}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::affine_grid_generator}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
