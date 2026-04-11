% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_hypot}
\alias{torch_hypot}
\title{Hypot}
\usage{
torch_hypot(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::hypot}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_hypot(torch_tensor(c(4.0)), torch_tensor(c(3.0, 4.0, 5.0)))
}
}
}
