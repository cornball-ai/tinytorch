% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cholesky}
\alias{torch_cholesky}
\title{Cholesky}
\usage{
torch_cholesky(self, upper = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{upper}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cholesky}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) # make symmetric positive-definite
l = torch_cholesky(a)
a
l
torch_mm(l, l$t())
a = torch_randn(c(3, 2, 2))

a = torch_matmul(a, a$transpose(-1, -2)) + 1e-03 # make symmetric positive-definite
l = torch_cholesky(a)
z = torch_matmul(l, l$transpose(-1, -2))
torch_max(torch_abs(z - a)) # Max non-zero

}
}
}
