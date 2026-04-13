% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_fractional_max_pool3d}
\alias{nnf_fractional_max_pool3d}
\title{Nnf fractional max pool3d}
\usage{
nnf_fractional_max_pool3d(self, kernel_size, output_size, random_samples)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{kernel_size}{Integer vector kernel size.}

\item{output_size}{Parameter passed to the underlying ATen operator.}

\item{random_samples}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf fractional max pool3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
