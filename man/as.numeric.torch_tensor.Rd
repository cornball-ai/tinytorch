% tinyrox says don't edit this manually, but it can't stop you!
\name{as.numeric.torch_tensor}
\alias{as.numeric.torch_tensor}
\title{As.numeric.torch tensor}
\usage{
\method{as.numeric}{torch_tensor}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\value{
An R object.
}
\description{
As.numeric.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_tensor(c(1, 2, 3))
as.numeric(x)
}
}
}
