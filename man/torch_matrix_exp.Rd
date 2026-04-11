% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_matrix_exp}
\alias{torch_matrix_exp}
\title{Matrix exp}
\usage{
torch_matrix_exp(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::matrix_exp}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_randn(c(2, 2, 2))
a[1, , ] <- torch_eye(2, 2)
a[2, , ] <- 2 * torch_eye(2, 2)
a
torch_matrix_exp(a)

x <- torch_tensor(rbind(c(0, pi/3), c(-pi/3, 0)))
x$matrix_exp() # should be [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
}
}
}
