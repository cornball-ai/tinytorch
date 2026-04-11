% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cholesky_inverse}
\alias{torch_cholesky_inverse}
\title{Cholesky inverse}
\usage{
torch_cholesky_inverse(self, upper = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{upper}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cholesky_inverse}.
}
\examples{
\donttest{
if (torch_is_installed()) {


a = torch_randn(c(3, 3))
a = torch_mm(a, a$t()) + 1e-05 * torch_eye(3) # make symmetric positive definite
u = torch_cholesky(a)
a
torch_cholesky_inverse(u)
a$inverse()

}
}
}
