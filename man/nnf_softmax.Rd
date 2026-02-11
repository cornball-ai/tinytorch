% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_softmax}
\alias{nnf_softmax}
\alias{torch_softmax}
\title{Softmax}
\usage{
nnf_softmax(self, dim = -1L)

torch_softmax(self, dim = -1L, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to apply softmax over (1-based).}

\item{dtype}{Ignored; for API compatibility.}
}
\description{
Softmax
}
\examples{
\donttest{
nnf_softmax(torch_randn(c(2, 3)), dim = 2)
}
}
