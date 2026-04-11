% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_diagonal}
\alias{torch_diagonal}
\title{Diagonal}
\usage{
torch_diagonal(self, offset = 0, dim1 = 0, dim2 = 1)
}
\arguments{
\item{self}{A torch_tensor.}

\item{offset}{An integer.}

\item{dim1}{An integer.}

\item{dim2}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::diagonal}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_diagonal(a, offset = 0)
torch_diagonal(a, offset = 1)
x = torch_randn(c(2, 5, 4, 2))
torch_diagonal(x, offset=-1, dim1=1, dim2=2)
}
}
}
