% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unique_consecutive}
\alias{torch_unique_consecutive}
\title{Unique consecutive}
\usage{
torch_unique_consecutive(self, return_inverse = FALSE, return_counts = FALSE,
                         dim = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{return_inverse}{A logical value.}

\item{return_counts}{A logical value.}

\item{dim}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unique_consecutive}.
}
