% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sgn}
\alias{torch_sgn}
\title{Sgn}
\usage{
torch_sgn(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::sgn}.
}
\examples{
\donttest{
if (torch_is_installed()) {
if (FALSE) {
x <- torch_tensor(c(3+4i, 7-24i, 0, 1+2i))
x$sgn()
torch_sgn(x)
}
}
}
}
