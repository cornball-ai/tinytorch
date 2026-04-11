% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_atan2}
\alias{torch_atan2}
\title{Atan2}
\usage{
torch_atan2(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::atan2}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_atan2(a, torch_randn(c(4)))
}
}
}
