% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pinverse}
\alias{torch_pinverse}
\title{Pinverse}
\usage{
torch_pinverse(self, rcond = 1e-15)
}
\arguments{
\item{self}{A torch_tensor.}

\item{rcond}{A numeric value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pinverse}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_randn(c(3, 5))
input
torch_pinverse(input)
# Batched pinverse example
a = torch_randn(c(2,6,3))
b = torch_pinverse(a)
torch_matmul(b, a)
}
}
}
