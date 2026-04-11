% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_mvlgamma}
\alias{torch_mvlgamma}
\title{Mvlgamma}
\usage{
torch_mvlgamma(self, p)
}
\arguments{
\item{self}{A torch_tensor.}

\item{p}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::mvlgamma}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_empty(c(2, 3))$uniform_(1, 2)
a
torch_mvlgamma(a, 2)
}
}
}
