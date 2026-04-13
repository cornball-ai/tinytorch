% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_trunc}
\alias{torch_trunc}
\title{torch_trunc}
\usage{
torch_trunc(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\value{
A `torch_tensor`.
}
\description{
torch_trunc
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_trunc(a)
}
}
}
