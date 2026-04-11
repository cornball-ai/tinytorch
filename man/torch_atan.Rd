% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_atan}
\alias{torch_atan}
\title{Atan}
\usage{
torch_atan(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::atan}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_atan(a)
}
}
}
