% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_log}
\alias{torch_log}
\title{Natural logarithm}
\usage{
torch_log(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Natural logarithm
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_log(a)
}
}
}
