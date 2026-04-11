% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_reciprocal}
\alias{torch_reciprocal}
\title{Reciprocal}
\usage{
torch_reciprocal(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::reciprocal}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_reciprocal(a)
}
}
}
