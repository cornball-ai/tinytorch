% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tan}
\alias{torch_tan}
\title{Tan}
\usage{
torch_tan(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tan}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_tan(a)
}
}
}
