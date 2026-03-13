% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fractional_max_pool3d}
\alias{torch_fractional_max_pool3d}
\title{Fractional max pool3d}
\usage{
torch_fractional_max_pool3d(self, kernel_size, output_size, random_samples)
}
\arguments{
\item{self}{A torch_tensor.}

\item{kernel_size}{(int[3]).}

\item{output_size}{(int[3]).}

\item{random_samples}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fractional_max_pool3d}.
}
