% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_unfold}
\alias{nnf_unfold}
\title{Nnf unfold}
\usage{
nnf_unfold(self, dimension, size, step)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dimension}{Parameter passed to the underlying ATen operator.}

\item{size}{Integer vector shape.}

\item{step}{Parameter passed to the underlying ATen operator.}
}
\value{
A `torch_tensor`.
}
\description{
Nnf unfold
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
