% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_vdot}
\alias{torch_vdot}
\title{Vdot}
\usage{
torch_vdot(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::vdot}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_vdot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))
if (FALSE) {
a <- torch_tensor(list(1 +2i, 3 - 1i))
b <- torch_tensor(list(2 +1i, 4 - 0i))
torch_vdot(a, b)
torch_vdot(b, a)
}
}
}
}
