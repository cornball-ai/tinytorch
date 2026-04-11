% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isclose}
\alias{torch_isclose}
\title{Isclose}
\usage{
torch_isclose(self, other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{rtol}{A numeric value.}

\item{atol}{A numeric value.}

\item{equal_nan}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isclose}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_isclose(torch_tensor(c(1., 2, 3)), torch_tensor(c(1 + 1e-10, 3, 4)))
torch_isclose(torch_tensor(c(Inf, 4)), torch_tensor(c(Inf, 6)), rtol=.5)
}
}
}
