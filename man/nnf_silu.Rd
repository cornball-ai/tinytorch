% tinyrox says don't edit this manually, but it can't stop you!
\name{nnf_silu}
\alias{nnf_silu}
\alias{torch_silu}
\title{SiLU activation (swish)}
\usage{
nnf_silu(self)

torch_silu(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\description{
SiLU activation (swish)
}
\examples{
\donttest{
nnf_silu(torch_randn(c(2, 3)))
}
}
