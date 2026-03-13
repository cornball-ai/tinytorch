% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rot90}
\alias{torch_rot90}
\title{Rot90}
\usage{
torch_rot90(self, k = 1, dims = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{k}{An integer.}

\item{dims}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rot90}.
}
