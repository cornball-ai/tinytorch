% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_square}
\alias{torch_square}
\title{Square}
\usage{
torch_square(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::square}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_square(a)
}
}
}
