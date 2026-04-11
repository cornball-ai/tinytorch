% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_as_strided}
\alias{torch_as_strided}
\title{As strided}
\usage{
torch_as_strided(self, size, stride, storage_offset = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{size}{An integer vector.}

\item{stride}{An integer vector.}

\item{storage_offset}{An integer. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::as_strided}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x = torch_randn(c(3, 3))
x
t = torch_as_strided(x, list(2, 2), list(1, 2))
t
t = torch_as_strided(x, list(2, 2), list(1, 2), 1)
t
}
}
}
