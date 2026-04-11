% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cholesky_solve}
\alias{torch_cholesky_solve}
\title{Cholesky solve}
\usage{
torch_cholesky_solve(self, input2, upper = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{input2}{A torch_tensor.}

\item{upper}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cholesky_solve}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) # make symmetric positive definite
u = torch_cholesky(a)
a
b = torch_randn(c(3, 2))
b
torch_cholesky_solve(b, u)
torch_mm(a$inverse(), b)
}
}
}
