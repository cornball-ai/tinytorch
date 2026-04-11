% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_atleast_2d}
\alias{torch_atleast_2d}
\title{Atleast 2d}
\usage{
torch_atleast_2d(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::atleast_2d}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_tensor(1.)
x
torch_atleast_2d(x)
x <- torch_randn(c(2,2))
x
torch_atleast_2d(x)
x <- torch_tensor(0.5)
y <- torch_tensor(1.)
torch_atleast_2d(list(x,y))
}
}
}
