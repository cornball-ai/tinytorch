% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_batch_norm3d}
\alias{nn_batch_norm3d}
\title{Nn batch norm3d}
\usage{
nn_batch_norm3d(num_features, eps = 1e-05, momentum = 0.1, affine = TRUE)
}
\arguments{
\item{num_features}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{momentum}{Numeric; momentum value.}

\item{affine}{Parameter passed to the underlying ATen operator.}
}
\value{
An `nn_module` instance.
}
\description{
Nn batch norm3d
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
