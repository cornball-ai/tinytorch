% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sort}
\alias{torch_sort}
\title{Sort tensor along a dimension}
\usage{
torch_sort(self, dim = -1L, descending = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to sort along. Default -1.}

\item{descending}{Whether to sort in descending order. Default FALSE.}
}
\value{
A list with values and indices tensors.
}
\description{
Sort tensor along a dimension
}
