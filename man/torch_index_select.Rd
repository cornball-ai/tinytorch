% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_index_select}
\alias{torch_index_select}
\title{torch_index_select}
\usage{
torch_index_select(self, dim, index)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}

\item{index}{A `torch_tensor` of indices.}
}
\description{
torch_index_select
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(3, 4))
x
indices = torch_tensor(c(1, 3), dtype = torch_int64())
torch_index_select(x, 1, indices)
torch_index_select(x, 2, indices)
}
}
}
