% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_layer_norm}
\alias{nn_layer_norm}
\title{Layer normalization module}
\usage{
nn_layer_norm(normalized_shape, eps = 1e-05)
}
\arguments{
\item{normalized_shape}{Integer or integer vector of the shape
to normalize over (typically the last dimension size).}

\item{eps}{Small constant for numerical stability.}
}
\value{
An nn_module instance.
}
\description{
Layer normalization module
}
