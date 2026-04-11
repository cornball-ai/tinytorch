% tinyrox says don't edit this manually, but it can't stop you!
\name{as.integer.torch_tensor}
\alias{as.integer.torch_tensor}
\title{As.integer.torch tensor}
\usage{
\method{as.integer}{torch_tensor}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\description{
As.integer.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_tensor(c(1, 2, 3))
as.integer(x)
}
}
}
