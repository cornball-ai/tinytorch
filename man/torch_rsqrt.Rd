% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rsqrt}
\alias{torch_rsqrt}
\title{Compute reciprocal square root of tensor}
\usage{
torch_rsqrt(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A `torch_tensor`.
}
\description{
Compute reciprocal square root of tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_rsqrt(torch_tensor(c(1, 4, 9)))
}
}
}
