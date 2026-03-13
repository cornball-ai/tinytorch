% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_new_zeros}
\alias{torch_new_zeros}
\title{New zeros}
\usage{
torch_new_zeros(self, size, dtype = NULL, device = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::new_zeros}.
}
