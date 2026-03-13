% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tril_indices}
\alias{torch_tril_indices}
\title{Tril indices}
\usage{
torch_tril_indices(row, col, offset = 0, dtype = long, device = NULL)
}
\arguments{
\item{row}{An integer.}

\item{col}{An integer.}

\item{offset}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tril_indices}.
}
