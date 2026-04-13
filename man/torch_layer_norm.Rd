% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_layer_norm}
\alias{torch_layer_norm}
\title{torch_layer_norm}
\usage{
torch_layer_norm(input, normalized_shape, weight = NULL, bias = NULL,
                 eps = 1e-05)
}
\arguments{
\item{input}{A `torch_tensor`.}

\item{normalized_shape}{Integer vector of normalized shape.}

\item{weight}{A `torch_tensor` of weights.}

\item{bias}{A `torch_tensor` of biases, or `NULL`.}

\item{eps}{Numeric; small constant for numerical stability.}
}
\value{
A `torch_tensor`.
}
\description{
torch_layer_norm
}
\examples{
\donttest{
if (torch_is_installed()) {
  # See PyTorch docs: https://docs.pytorch.org/docs/stable/torch.html
}
}
}
