% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_triu}
\alias{torch_triu}
\title{Upper triangular part of a matrix}
\usage{
torch_triu(self, diagonal = 0L)
}
\arguments{
\item{self}{A torch_tensor.}

\item{diagonal}{Offset from the main diagonal. Default 0.}
}
\value{
A torch_tensor.
}
\description{
Upper triangular part of a matrix
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_triu(a)
torch_triu(a, diagonal=1)
torch_triu(a, diagonal=-1)
b = torch_randn(c(4, 6))
b
torch_triu(b, diagonal=1)
torch_triu(b, diagonal=-1)
}
}
}
