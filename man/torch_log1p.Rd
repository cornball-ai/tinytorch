% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_log1p}
\alias{torch_log1p}
\title{Log1p}
\usage{
torch_log1p(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::log1p}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(5))
a
torch_log1p(a)
}
}
}
