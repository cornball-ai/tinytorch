% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diagflat}
\alias{torch_diagflat}
\title{Diagflat}
\usage{
torch_diagflat(self, offset = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{offset}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diagflat}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3))
a
torch_diagflat(a)
torch_diagflat(a, 1)
a = torch_randn(c(2, 2))
a
torch_diagflat(a)
}
}
}
