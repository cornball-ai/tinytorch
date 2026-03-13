% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lu_unpack}
\alias{torch_lu_unpack}
\title{Lu unpack}
\usage{
torch_lu_unpack(LU_data, LU_pivots, unpack_data = TRUE, unpack_pivots = TRUE)
}
\arguments{
\item{LU_data}{A torch_tensor.}

\item{LU_pivots}{A torch_tensor.}

\item{unpack_data}{A logical value.}

\item{unpack_pivots}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lu_unpack}.
}
