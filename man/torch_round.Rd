% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_round}
\alias{torch_round}
\title{torch_round}
\usage{
torch_round(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\description{
torch_round
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_round(a)
}
}
}
