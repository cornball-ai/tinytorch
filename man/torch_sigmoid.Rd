% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_sigmoid}
\alias{torch_sigmoid}
\title{Sigmoid activation (namespace-level)}
\usage{
torch_sigmoid(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\description{
Sigmoid activation (namespace-level)
}
\examples{
\donttest{
if (torch_is_installed()) {
x <- torch_randn(c(2, 3))
torch_sigmoid(x)
}
}
}
