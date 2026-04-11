% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sign}
\alias{torch_sign}
\title{torch_sign}
\usage{
torch_sign(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\description{
torch_sign
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_tensor(c(0.7, -1.2, 0., 2.3))
a
torch_sign(a)
}
}
}
