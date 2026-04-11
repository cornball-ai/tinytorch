% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logaddexp}
\alias{torch_logaddexp}
\title{Logaddexp}
\usage{
torch_logaddexp(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logaddexp}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_logaddexp(torch_tensor(c(-1.0)), torch_tensor(c(-1.0, -2, -3)))
torch_logaddexp(torch_tensor(c(-100.0, -200, -300)), torch_tensor(c(-1.0, -2, -3)))
torch_logaddexp(torch_tensor(c(1.0, 2000, 30000)), torch_tensor(c(-1.0, -2, -3)))
}
}
}
