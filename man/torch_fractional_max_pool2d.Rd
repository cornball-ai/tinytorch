% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fractional_max_pool2d}
\alias{torch_fractional_max_pool2d}
\title{Fractional max pool2d}
\usage{
torch_fractional_max_pool2d(self, kernel_size, output_size, random_samples)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[2]).}

\item{output_size}{(int[2]).}

\item{random_samples}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fractional_max_pool2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
