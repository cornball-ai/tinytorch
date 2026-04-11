% tinyrox says don't edit this manually, but it can't stop you!
\name{length.nn_module}
\alias{length.nn_module}
\title{Length.nn module}
\usage{
\method{length}{nn_module}(x)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}
}
\description{
Length.nn module
}
\examples{
\donttest{
if (torch_is_installed()) {
length(nn_linear(10, 5))
}
}
}
