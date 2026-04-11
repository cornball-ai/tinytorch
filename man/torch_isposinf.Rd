% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isposinf}
\alias{torch_isposinf}
\title{Isposinf}
\usage{
torch_isposinf(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isposinf}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(-Inf, Inf, 1.2))
torch_isposinf(a)
}
}
}
