% tinyrox says don't edit this manually, but it can't stop you!
\name{length.nn_buffer}
\alias{length.nn_buffer}
\title{Length.nn buffer}
\usage{
\method{length}{nn_buffer}(x)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}
}
\description{
Length.nn buffer
}
\examples{
\donttest{
if (torch_is_installed()) {
length(nn_buffer(torch_randn(5)))
}
}
}
