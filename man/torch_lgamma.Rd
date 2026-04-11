% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_lgamma}
\alias{torch_lgamma}
\title{Lgamma}
\usage{
torch_lgamma(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::lgamma}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_arange(0.5, 2, 0.5)
torch_lgamma(a)
}
}
}
