% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_atleast_1d}
\alias{torch_atleast_1d}
\title{Atleast 1d}
\usage{
torch_atleast_1d(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::atleast_1d}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_randn(c(2))
x
torch_atleast_1d(x)
x <- torch_tensor(1.)
x
torch_atleast_1d(x)
x <- torch_tensor(0.5)
y <- torch_tensor(1.)
torch_atleast_1d(list(x,y))
}
}
}
