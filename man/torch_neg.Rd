% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_neg}
\alias{torch_neg}
\title{torch_neg}
\usage{
torch_neg(self)
}
\arguments{
\item{self}{A `torch_tensor`.}
}
\description{
torch_neg
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_neg(a)
}
}
}
