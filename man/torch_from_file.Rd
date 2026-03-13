% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_from_file}
\alias{torch_from_file}
\title{From file}
\usage{
torch_from_file(filename, shared = NULL, size = 0, dtype = NULL, device = NULL)
}
\arguments{
\item{filename}{A character string.}

\item{shared}{A logical value. Optional.}

\item{size}{An integer. Optional.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::from_file}.
}
