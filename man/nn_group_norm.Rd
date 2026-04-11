% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_group_norm}
\alias{nn_group_norm}
\title{Nn group norm}
\usage{
nn_group_norm(num_groups, num_channels, eps = 1e-05, affine = TRUE)
}
\arguments{
\item{num_groups}{Parameter passed to the underlying ATen operator.}

\item{num_channels}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{affine}{Parameter passed to the underlying ATen operator.}
}
\description{
Nn group norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
