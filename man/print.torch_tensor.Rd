% tinyrox says don't edit this manually, but it can't stop you!
\name{print.torch_tensor}
\alias{print.torch_tensor}
\title{Print.torch tensor}
\usage{
\method{print}{torch_tensor}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\description{
Print.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
print(x)
}
}
}
