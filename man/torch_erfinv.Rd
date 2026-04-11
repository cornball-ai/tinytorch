% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_erfinv}
\alias{torch_erfinv}
\title{Erfinv}
\usage{
torch_erfinv(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::erfinv}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_erfinv(torch_tensor(c(0, 0.5, -1.)))
}
}
}
