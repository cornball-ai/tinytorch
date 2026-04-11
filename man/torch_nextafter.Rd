% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nextafter}
\alias{torch_nextafter}
\title{Nextafter}
\usage{
torch_nextafter(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nextafter}.
}
\examples{
\donttest{
if (torch_is_installed()) {

eps <- torch_finfo(torch_float32())$eps
torch_nextafter(torch_tensor(c(1, 2)), torch_tensor(c(2, 1))) == torch_tensor(c(eps + 1, 2 - eps))
}
}
}
