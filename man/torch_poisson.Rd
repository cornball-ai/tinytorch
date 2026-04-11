% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_poisson}
\alias{torch_poisson}
\title{Poisson}
\usage{
torch_poisson(self, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::poisson}.
}
\examples{
\donttest{
if (torch_is_installed()) {

rates = torch_rand(c(4, 4)) * 5  # rate parameter between 0 and 5
torch_poisson(rates)
}
}
}
