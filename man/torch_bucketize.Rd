% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bucketize}
\alias{torch_bucketize}
\title{Bucketize}
\usage{
torch_bucketize(self, boundaries, out_int32 = FALSE, right = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{boundaries}{A torch_tensor.}

\item{out_int32}{A logical value.}

\item{right}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bucketize}.
}
