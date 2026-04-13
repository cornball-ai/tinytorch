% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_max_unpool3d}
\alias{nnf_max_unpool3d}
\title{Nnf max unpool3d}
\usage{
nnf_max_unpool3d(self, indices, output_size, stride, padding)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{indices}{A `torch_tensor` of indices.}

\item{output_size}{Parameter passed to the underlying ATen operator.}

\item{stride}{Integer vector stride.}

\item{padding}{Integer vector padding.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf max unpool3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
