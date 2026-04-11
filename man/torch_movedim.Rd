% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_movedim}
\alias{torch_movedim}
\title{Movedim}
\usage{
torch_movedim(self, source, destination)
}
\arguments{
\item{self}{A torch_tensor.}

\item{source}{An integer vector.}

\item{destination}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::movedim}.
}
\examples{
\donttest{
if (torch_is_installed()) {

t <- torch_randn(c(3,2,1))
t
torch_movedim(t, 2, 1)$shape
torch_movedim(t, 2, 1)
torch_movedim(t, c(2, 3), c(1, 2))$shape
torch_movedim(t, c(2, 3), c(1, 2))
}
}
}
