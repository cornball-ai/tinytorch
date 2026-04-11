% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_count_nonzero}
\alias{torch_count_nonzero}
\title{Count nonzero}
\usage{
torch_count_nonzero(self, dim)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::count_nonzero}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_zeros(3,3)
x[torch_randn(3,3) > 0.5] = 1
x
torch_count_nonzero(x)
torch_count_nonzero(x, dim=1)
}
}
}
