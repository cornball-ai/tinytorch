% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_digamma}
\alias{torch_digamma}
\title{Digamma}
\usage{
torch_digamma(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::digamma}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_tensor(c(1, 0.5))
torch_digamma(a)
}
}
}
