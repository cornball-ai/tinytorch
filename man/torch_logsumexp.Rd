% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logsumexp}
\alias{torch_logsumexp}
\title{Logsumexp}
\usage{
torch_logsumexp(self, dim, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]).}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logsumexp}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(3, 3))
torch_logsumexp(a, 1)
}
}
}
