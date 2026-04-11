% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logdet}
\alias{torch_logdet}
\title{Logdet}
\usage{
torch_logdet(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logdet}.
}
\examples{
\donttest{
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
torch_det(A)
torch_logdet(A)
A
A$det()
A$det()$log()
}
}
}
