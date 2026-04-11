% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_prod}
\alias{torch_prod}
\title{Prod}
\usage{
torch_prod(self, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::prod}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_prod(a)


a = torch_randn(c(4, 2))
a
torch_prod(a, 1)
}
}
}
