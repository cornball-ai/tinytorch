% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_outer}
\alias{torch_outer}
\title{Outer product of two vectors}
\usage{
torch_outer(self, vec2)
}
\arguments{
\item{self}{First vector.}

\item{vec2}{Second vector.}
}
\value{
A torch_tensor.
}
\description{
Outer product of two vectors
}
\examples{
\donttest{
if (torch_is_installed()) {

v1 <- torch_arange(1., 5.)
v2 <- torch_arange(1., 4.)
torch_outer(v1, v2)
}
}
}
