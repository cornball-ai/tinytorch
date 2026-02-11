% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_norm}
\alias{torch_norm}
\title{Norm of a tensor}
\usage{
torch_norm(self, p = 2, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{Norm order. Default 2.}

\item{dim}{Optional dimension to reduce.}

\item{keepdim}{Whether to keep the reduced dimension.}
}
\value{
A torch_tensor.
}
\description{
Norm of a tensor
}
