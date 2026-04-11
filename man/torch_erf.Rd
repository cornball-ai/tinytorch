% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_erf}
\alias{torch_erf}
\title{Erf}
\usage{
torch_erf(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::erf}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_erf(torch_tensor(c(0, -1., 10.)))
}
}
}
