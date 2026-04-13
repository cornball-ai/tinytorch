% tinyrox says don't edit this manually, but it can't stop you!
\name{length.nn_parameter}
\alias{length.nn_parameter}
\title{Length.nn parameter}
\usage{
\method{length}{nn_parameter}(x)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}
}
\value{
Integer scalar.
}
\description{
Length.nn parameter
}
\examples{
\donttest{
if (torch_is_installed()) {
length(nn_parameter(torch_randn(5)))
}
}
}
