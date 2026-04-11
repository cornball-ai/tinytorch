% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isnan}
\alias{torch_isnan}
\title{Isnan}
\usage{
torch_isnan(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isnan}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_isnan(torch_tensor(c(1, NaN, 2)))
}
}
}
