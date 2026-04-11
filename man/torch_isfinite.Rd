% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_isfinite}
\alias{torch_isfinite}
\title{Isfinite}
\usage{
torch_isfinite(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::isfinite}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_isfinite(torch_tensor(c(1, Inf, 2, -Inf, NaN)))
}
}
}
