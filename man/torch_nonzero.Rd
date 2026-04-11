% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nonzero}
\alias{torch_nonzero}
\title{Nonzero}
\usage{
torch_nonzero(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nonzero}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_nonzero(torch_tensor(c(1, 1, 1, 0, 1)))
}
}
}
