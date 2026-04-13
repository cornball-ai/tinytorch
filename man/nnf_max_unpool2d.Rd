% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_max_unpool2d}
\alias{nnf_max_unpool2d}
\title{Nnf max unpool2d}
\usage{
nnf_max_unpool2d(self, indices, output_size)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{indices}{A `torch_tensor` of indices.}

\item{output_size}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf max unpool2d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
