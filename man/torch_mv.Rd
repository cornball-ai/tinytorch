% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mv}
\alias{torch_mv}
\title{Mv}
\usage{
torch_mv(self, vec)
}
\arguments{
\item{self}{A torch_tensor.}

\item{vec}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::mv}.
}
\examples{
\donttest{
if (torch_is_installed()) {

mat = torch_randn(c(2, 3))
vec = torch_randn(c(3))
torch_mv(mat, vec)
}
}
}
