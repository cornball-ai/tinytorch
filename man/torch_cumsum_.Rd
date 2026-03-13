% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cumsum_}
\alias{torch_cumsum_}
\title{Cumsum}
\usage{
torch_cumsum_(self, dim, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cumsum_}.
}
