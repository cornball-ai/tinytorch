% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_exp}
\alias{torch_exp}
\title{Exponential}
\usage{
torch_exp(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Exponential
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_exp(torch_tensor(c(0, log(2))))
}
}
}
