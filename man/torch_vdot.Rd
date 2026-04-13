% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_vdot}
\alias{torch_vdot}
\title{Vdot}
\usage{
torch_vdot(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::vdot}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_vdot(torch_tensor(c(2, 3)), torch_tensor(c(2, 1)))

}
}
}
