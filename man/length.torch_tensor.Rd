% tinyrox says don't edit this manually, but it can't stop you!
\name{length.torch_tensor}
\alias{length.torch_tensor}
\title{Length.torch tensor}
\usage{
\method{length}{torch_tensor}(x)
}
\arguments{
\item{x}{Parameter passed to the underlying ATen operator.}
}
\description{
Length.torch tensor
}
\examples{
\donttest{
if (torch_is_installed()) {
length(torch_randn(c(2, 3)))
}
}
}
