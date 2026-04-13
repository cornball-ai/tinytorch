% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_avg_pool3d}
\alias{nnf_avg_pool3d}
\title{Nnf avg pool3d}
\usage{
nnf_avg_pool3d(self, kernel_size, stride, padding, ceil_mode,
               count_include_pad, divisor_override)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{kernel_size}{Integer vector kernel size.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}

\item{ceil_mode}{Parameter passed to the underlying ATen operator.}

\item{count_include_pad}{Parameter passed to the underlying ATen operator.}

\item{divisor_override}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf avg pool3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
