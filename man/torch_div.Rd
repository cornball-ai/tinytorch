% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_div}
\alias{torch_div}
\title{Divide two tensors element-wise}
\usage{
torch_div(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor or scalar.}
}
\description{
Divide two tensors element-wise
}
\examples{
\dontrun{
a <- torch_randn(c(2, 3))
torch_div(a, 2.0)
}
}
