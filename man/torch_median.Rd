% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_median}
\alias{torch_median}
\title{Median}
\usage{
torch_median(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::median}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_median(a)


a = torch_randn(c(4, 5))
a
torch_median(a, 1)
}
}
}
