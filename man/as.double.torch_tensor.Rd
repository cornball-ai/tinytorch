% tinyrox says don't edit this manually, but it can't stop you!
\name{as.double.torch_tensor}
\alias{as.double.torch_tensor}
\title{As.double.torch tensor}
\usage{
\method{as.double}{torch_tensor}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\description{
As.double.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_tensor(c(1, 2, 3))
as.double(x)
}
}
}
