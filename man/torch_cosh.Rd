% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_cosh}
\alias{torch_cosh}
\title{Cosh}
\usage{
torch_cosh(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::cosh}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_cosh(a)
}
}
}
