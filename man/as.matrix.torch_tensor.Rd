% tinyrox says don't edit this manually, but it can't stop you!
\name{as.matrix.torch_tensor}
\alias{as.matrix.torch_tensor}
\title{As.matrix.torch tensor}
\usage{
\method{as.matrix}{torch_tensor}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\value{
An R object.
}
\description{
As.matrix.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_tensor(matrix(1:6, 2, 3))
as.matrix(x)
}
}
}
