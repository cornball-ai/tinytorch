% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rad2deg}
\alias{torch_rad2deg}
\title{Rad2deg}
\usage{
torch_rad2deg(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rad2deg}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(rbind(c(3.142, -3.142), c(6.283, -6.283), c(1.570, -1.570)))
torch_rad2deg(a)
}
}
}
