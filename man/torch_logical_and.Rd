% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logical_and}
\alias{torch_logical_and}
\title{Logical and}
\usage{
torch_logical_and(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logical_and}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_logical_and(torch_tensor(c(TRUE, FALSE, TRUE)), torch_tensor(c(TRUE, FALSE, FALSE)))
a = torch_tensor(c(0, 1, 10, 0), dtype=torch_int8())
b = torch_tensor(c(4, 0, 1, 0), dtype=torch_int8())
torch_logical_and(a, b)

torch_logical_and(a, b, out=torch_empty(4, dtype=torch_bool()))

}
}
}
