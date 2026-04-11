% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_frac}
\alias{torch_frac}
\title{Frac}
\usage{
torch_frac(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::frac}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_frac(torch_tensor(c(1, 2.5, -3.2)))
}
}
}
