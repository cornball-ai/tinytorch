% tinyrox says don't edit this manually, but it can't stop you!
\name{nn_sequential}
\alias{nn_sequential}
\title{Sequential container}
\usage{
nn_sequential(...)
}
\arguments{
\item{...}{nn_module instances to chain sequentially.}
}
\value{
An nn_module instance.
}
\description{
Sequential container
}
\examples{
\dontrun{
m <- nn_sequential(nn_linear(3, 4), nn_relu(), nn_linear(4, 2))
m(torch_randn(c(1, 3)))
}
}
