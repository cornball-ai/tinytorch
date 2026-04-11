% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_det}
\alias{torch_det}
\title{Det}
\usage{
torch_det(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::det}.
}
\examples{
\donttest{
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
torch_det(A)
A = torch_randn(c(3, 2, 2))
A
A$det()
}
}
}
