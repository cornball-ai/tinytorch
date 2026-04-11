% tinyrox says don't edit this manually, but it can't stop you!
\name{distr_gamma}
\alias{distr_gamma}
\title{Gamma distribution}
\usage{
distr_gamma(concentration, rate)
}
\arguments{
\item{concentration}{Shape parameter (tensor or numeric).}

\item{rate}{Rate parameter (tensor or numeric).}
}
\value{
A distribution object.
}
\description{
Gamma distribution
}
\examples{
\donttest{
if (torch_is_installed()) {
m <- distr_gamma(torch_tensor(1.0), torch_tensor(1.0))
m$sample() # Gamma distributed with concentration=1 and rate=1
}
}
}
