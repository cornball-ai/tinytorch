% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_argsort}
\alias{torch_argsort}
\title{Argsort}
\usage{
torch_argsort(self, dim = -1, descending = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{descending}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::argsort}.
}
