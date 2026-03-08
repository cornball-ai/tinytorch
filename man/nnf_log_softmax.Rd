% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_log_softmax}
\alias{nnf_log_softmax}
\alias{torch_log_softmax}
\title{Log-softmax}
\usage{
nnf_log_softmax(self, dim = -1L)

torch_log_softmax(self, dim = -1L, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to apply log-softmax over (1-based).}

\item{dtype}{Ignored; for API compatibility.}
}
\description{
Log-softmax
}
\examples{
\dontrun{
nnf_log_softmax(torch_randn(c(2, 3)), dim = 2)
}
}
