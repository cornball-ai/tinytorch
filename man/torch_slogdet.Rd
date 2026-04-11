% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_slogdet}
\alias{torch_slogdet}
\title{Slogdet}
\usage{
torch_slogdet(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::slogdet}.
}
\examples{
\donttest{
if (torch_is_installed()) {

A = torch_randn(c(3, 3))
A
torch_det(A)
torch_logdet(A)
torch_slogdet(A)
}
}
}
