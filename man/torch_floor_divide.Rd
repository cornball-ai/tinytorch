% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_floor_divide}
\alias{torch_floor_divide}
\title{torch_floor_divide}
\usage{
torch_floor_divide(self, other)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{other}{A `torch_tensor` or scalar.}
}
\description{
torch_floor_divide
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_tensor(c(4.0, 3.0))
b = torch_tensor(c(2.0, 2.0))
torch_floor_divide(a, b)
torch_floor_divide(a, 1.4)
}
}
}
