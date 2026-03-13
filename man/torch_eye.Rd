% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_eye}
\alias{torch_eye}
\title{Eye}
\usage{
torch_eye(n, dtype = NULL, device = NULL)
}
\arguments{
\item{n}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::eye}.
}
