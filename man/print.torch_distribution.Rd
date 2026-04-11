% tinyrox says don't edit this manually, but it can't stop you!
\name{print.torch_distribution}
\alias{print.torch_distribution}
\title{Print.torch distribution}
\usage{
\method{print}{torch_distribution}(x, ...)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}

\item{...}{Additional arguments passed to methods.}
}
\description{
Print.torch distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
d <- distr_normal(0, 1)
print(d)
}
}
}
