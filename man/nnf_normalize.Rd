% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_normalize}
\alias{nnf_normalize}
\title{Normalize a tensor along a dimension}
\usage{
nnf_normalize(input, p = 2, dim = -1L, eps = 1e-12)
}
\arguments{
\item{input}{A torch_tensor.}

\item{p}{Norm order. Default 2.}

\item{dim}{Dimension to normalize along. Default -1.}

\item{eps}{Small constant for numerical stability. Default 1e-12.}
}
\value{
A torch_tensor.
}
\description{
Performs Lp normalization along the given dimension.
}
\examples{
\donttest{
if (torch_is_installed()) {
nnf_normalize(torch_randn(c(2, 3)))
}
}
}
