% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_pairwise_distance}
\alias{nn_pairwise_distance}
\title{Nn pairwise distance}
\usage{
nn_pairwise_distance(p = 2, eps = 1e-06, keepdim = FALSE)
}
\arguments{
\item{p}{Parameter passed to the underlying ATen operator.}

\item{eps}{Numeric; small constant for numerical stability.}

\item{keepdim}{Logical; keep reduced dimensions.}
}
\value{
An `nn_module` instance.
}
\description{
Nn pairwise distance
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
