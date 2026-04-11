% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_atanh}
\alias{torch_atanh}
\title{Atanh}
\usage{
torch_atanh(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::atanh}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))$uniform_(-1, 1)
a
torch_atanh(a)
}
}
}
