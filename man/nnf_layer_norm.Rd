% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_layer_norm}
\alias{nnf_layer_norm}
\alias{torch_layer_norm}
\title{Layer normalization}
\usage{
nnf_layer_norm(input, normalized_shape, weight = NULL, bias = NULL, eps = 1e-05)

torch_layer_norm(input, normalized_shape, weight = NULL, bias = NULL,
                 eps = 1e-05)
}
\arguments{
\item{input}{A torch_tensor.}

\item{normalized_shape}{Integer vector of shape to normalize.}

\item{weight}{Optional weight tensor.}

\item{bias}{Optional bias tensor.}

\item{eps}{Small constant for numerical stability.}
}
\description{
Layer normalization
}
\examples{
\donttest{
x <- torch_randn(c(2, 4))
nnf_layer_norm(x, 4L)
}
}
