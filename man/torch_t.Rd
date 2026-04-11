% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_t}
\alias{torch_t}
\title{torch_t}
\usage{
torch_t(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\description{
torch_t
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(2,3))
x
torch_t(x)
x = torch_randn(c(3))
x
torch_t(x)
x = torch_randn(c(2, 3))
x
torch_t(x)
}
}
}
