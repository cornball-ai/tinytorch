% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isneginf}
\alias{torch_isneginf}
\title{Isneginf}
\usage{
torch_isneginf(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isneginf}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(-Inf, Inf, 1.2))
torch_isneginf(a)
}
}
}
