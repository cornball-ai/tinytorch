% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isinf}
\alias{torch_isinf}
\title{Isinf}
\usage{
torch_isinf(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isinf}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_isinf(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
}
}
}
