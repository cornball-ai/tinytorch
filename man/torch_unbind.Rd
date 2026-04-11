% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_unbind}
\alias{torch_unbind}
\title{Unbind}
\usage{
torch_unbind(self, dim = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::unbind}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_unbind(torch_tensor(matrix(1:9, ncol = 3, byrow=TRUE)))
}
}
}
