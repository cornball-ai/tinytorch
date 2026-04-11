% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isreal}
\alias{torch_isreal}
\title{Isreal}
\usage{
torch_isreal(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isreal}.
}
\examples{
\donttest{
if (torch_is_installed()) {
if (FALSE) {
torch_isreal(torch_tensor(c(1, 1+1i, 2+0i)))
}
}
}
}
