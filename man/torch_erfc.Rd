% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_erfc}
\alias{torch_erfc}
\title{Erfc}
\usage{
torch_erfc(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::erfc}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_erfc(torch_tensor(c(0, -1., 10.)))
}
}
}
