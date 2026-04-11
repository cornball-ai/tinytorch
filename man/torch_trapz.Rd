% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_trapz}
\alias{torch_trapz}
\title{Trapz}
\usage{
torch_trapz(y, x, dim = -1)
}
\arguments{
\item{y}{A torch_tensor.}

\item{x}{A torch_tensor.}

\item{dim}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::trapz}.
}
\examples{
\donttest{
if (torch_is_installed()) {

y = torch_randn(list(2, 3))
y
x = torch_tensor(matrix(c(1, 3, 4, 1, 2, 3), ncol = 3, byrow=TRUE))
torch_trapz(y, x = x)

}
}
}
