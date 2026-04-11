% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_fmod}
\alias{torch_fmod}
\title{Fmod}
\usage{
torch_fmod(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::fmod}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_fmod(torch_tensor(c(-3., -2, -1, 1, 2, 3)), 2)
torch_fmod(torch_tensor(c(1., 2, 3, 4, 5)), 1.5)
}
}
}
