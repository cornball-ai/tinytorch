% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_matmul}
\alias{torch_matmul}
\title{Matrix multiplication}
\usage{
torch_matmul(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\description{
Matrix multiplication
}
\examples{
\donttest{
a <- torch_randn(c(2, 3))
b <- torch_randn(c(3, 4))
torch_matmul(a, b)
}
}
