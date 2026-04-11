% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_trace}
\alias{torch_trace}
\title{Trace}
\usage{
torch_trace(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::trace}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_arange(1, 9)$view(c(3, 3))
x
torch_trace(x)
}
}
}
