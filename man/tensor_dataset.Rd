% tinyrox says don't edit this manually, but it can't stop you!
\name{tensor_dataset}
\alias{tensor_dataset}
\title{Create a dataset from tensors}
\usage{
tensor_dataset(...)
}
\arguments{
\item{...}{Tensors. Each must have the same first dimension.}
}
\value{
A dataset object.
}
\description{
Create a dataset from tensors
}
\examples{
\donttest{
if (torch_is_installed()) {
ds <- tensor_dataset(torch_randn(c(10, 3)), torch_randint(2L, 10L))
length(ds)
}
}
}
