% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unique_dim}
\alias{torch_unique_dim}
\title{Unique dim}
\usage{
torch_unique_dim(self, dim, sorted = TRUE, return_inverse = FALSE,
                 return_counts = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}

\item{sorted}{A logical value.}

\item{return_inverse}{A logical value.}

\item{return_counts}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unique_dim}.
}
