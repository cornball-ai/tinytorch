% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_init_sparse_}
\alias{nn_init_sparse_}
\title{Sparse initialization}
\usage{
nn_init_sparse_(tensor, sparsity, std = 0.01)
}
\arguments{
\item{tensor}{A 2D tensor.}

\item{sparsity}{Fraction of elements to be zero in each column.}

\item{std}{Standard deviation of the normal distribution. Default 0.01.}
}
\value{
tensor (modified in place).
}
\description{
Sparse initialization
}
\examples{
\donttest{
if (torch_is_installed()) {

w <- torch_empty(3, 5)
nn_init_sparse_(w, sparsity = 0.1)

}
}
}
