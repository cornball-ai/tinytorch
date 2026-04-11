% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_equal}
\alias{torch_equal}
\title{Equal}
\usage{
torch_equal(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::equal}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_equal(torch_tensor(c(1, 2)), torch_tensor(c(1, 2)))
}
}
}
