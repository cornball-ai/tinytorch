% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_amin}
\alias{torch_amin}
\title{Amin}
\usage{
torch_amin(self, dim = NULL, keepdim = FALSE)
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
Wrapper for \code{at::amin}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_randn(c(4, 4))
a
torch_amin(a, 1)
}
}
}
