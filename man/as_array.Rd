% tinyrox says don't edit this manually, but it can't stop you!
\name{as_array}
\alias{as_array}
\title{Convert a tensor to an R array}
\usage{
as_array(x)
}
\arguments{
\item{x}{A torch_tensor.}
}
\value{
An R numeric vector, matrix, or array.
}
\description{
Convert a tensor to an R array
}
\examples{
\donttest{
x <- torch_tensor(c(1, 2, 3))
as_array(x)
}
}
