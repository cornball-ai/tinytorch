% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tril}
\alias{torch_tril}
\title{Tril}
\usage{
torch_tril(self, diagonal = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{diagonal}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tril}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a
torch_tril(a)
b = torch_randn(c(4, 6))
b
torch_tril(b, diagonal=1)
torch_tril(b, diagonal=-1)
}
}
}
