% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ceil}
\alias{torch_ceil}
\title{torch_ceil}
\usage{
torch_ceil(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\value{
A `torch_tensor`.
}
\description{
torch_ceil
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_ceil(a)
}
}
}
