% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_max_unpool3d}
\alias{torch_max_unpool3d}
\title{Max unpool3d}
\usage{
torch_max_unpool3d(self, indices, output_size, stride, padding)
}
\arguments{
\item{self}{A torch_tensor.}

\item{indices}{A torch_tensor.}

\item{output_size}{(int[3]).}

\item{stride}{(int[3]).}

\item{padding}{(int[3]).}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::max_unpool3d}.
}
