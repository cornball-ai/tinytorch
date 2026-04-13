% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_softmax}
\alias{nnf_softmax}
\title{Softmax}
\usage{
nnf_softmax(self, dim = -1L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to apply softmax over (1-based).}
}
\value{
A `torch_tensor`.
}
\description{
Softmax
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_softmax(torch_randn(c(2, 3)), dim = 2)
}
}
}
