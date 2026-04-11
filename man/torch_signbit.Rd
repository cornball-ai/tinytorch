% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_signbit}
\alias{torch_signbit}
\title{Signbit}
\usage{
torch_signbit(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::signbit}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(0.7, -1.2, 0., 2.3))
torch_signbit(a)
}
}
}
