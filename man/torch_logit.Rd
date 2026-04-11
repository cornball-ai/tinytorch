% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_logit}
\alias{torch_logit}
\title{Logit}
\usage{
torch_logit(self, eps = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{eps}{A numeric value. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::logit}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_rand(5)
a
torch_logit(a, eps=1e-6)
}
}
}
