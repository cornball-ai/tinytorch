% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_view_as_complex}
\alias{torch_view_as_complex}
\title{View as complex}
\usage{
torch_view_as_complex(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::view_as_complex}.
}
\examples{
\donttest{
if (torch_is_installed()) {
if (FALSE) {
x=torch_randn(c(4, 2))
x
torch_view_as_complex(x)
}
}
}
}
