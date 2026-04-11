% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fliplr}
\alias{torch_fliplr}
\title{Fliplr}
\usage{
torch_fliplr(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fliplr}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_arange(start = 1, end = 4)$view(c(2, 2))
x
torch_fliplr(x)
}
}
}
