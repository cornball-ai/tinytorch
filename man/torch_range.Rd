% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_range}
\alias{torch_range}
\title{Range}
\usage{
torch_range(start, end, step = 1, dtype = NULL, device = NULL)
}
\arguments{
\item{start}{A numeric scalar.}

\item{end}{A numeric scalar.}

\item{step}{A numeric scalar.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::range}.
}
