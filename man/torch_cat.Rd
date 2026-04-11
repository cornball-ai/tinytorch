% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cat}
\alias{torch_cat}
\title{Concatenate tensors along a dimension}
\usage{
torch_cat(tensors, dim = 1L)
}
\arguments{
\item{tensors}{A list of torch_tensors.}

\item{dim}{Dimension to concatenate along.}
}
\value{
A torch_tensor.
}
\description{
Concatenate tensors along a dimension
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(2, 3))
x
torch_cat(list(x, x, x), 1)
torch_cat(list(x, x, x), 2)
}
}
}
