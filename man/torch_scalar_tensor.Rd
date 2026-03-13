% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_scalar_tensor}
\alias{torch_scalar_tensor}
\title{Scalar tensor}
\usage{
torch_scalar_tensor(s, dtype = NULL, device = NULL)
}
\arguments{
\item{s}{A numeric scalar.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::scalar_tensor}.
}
