% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rms_norm}
\alias{torch_rms_norm}
\title{Rms norm}
\usage{
torch_rms_norm(input, normalized_shape, weight = NULL, eps = NULL)
}
\arguments{
\item{input}{A torch_tensor.}

\item{normalized_shape}{An integer vector.}

\item{weight}{A torch_tensor. Optional.}

\item{eps}{A numeric value. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rms_norm}.
}
