% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_empty_like}
\alias{torch_empty_like}
\title{Create an uninitialized tensor with same shape/dtype as input}
\usage{
torch_empty_like(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A `torch_tensor`.
}
\description{
Create an uninitialized tensor with same shape/dtype as input
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
torch_empty_like(x)
}
}
}
