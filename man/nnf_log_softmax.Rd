% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_log_softmax}
\alias{nnf_log_softmax}
\title{Log-softmax}
\usage{
nnf_log_softmax(self, dim = -1L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Dimension to apply log-softmax over (1-based).}
}
\value{
A `torch_tensor`.
}
\description{
Log-softmax
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_log_softmax(torch_randn(c(2, 3)), dim = 2)
}
}
}
