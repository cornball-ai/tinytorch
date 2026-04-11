% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_poisson}
\alias{distr_poisson}
\title{Poisson distribution}
\usage{
distr_poisson(rate)
}
\arguments{
\item{rate}{Rate parameter (tensor or numeric).}
}
\value{
A distribution object.
}
\description{
Poisson distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_poisson(torch_tensor(4))
m$sample()
}
}
}
