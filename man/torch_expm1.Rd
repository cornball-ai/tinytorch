% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_expm1}
\alias{torch_expm1}
\title{Expm1}
\usage{
torch_expm1(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::expm1}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_expm1(torch_tensor(c(0, log(2))))
}
}
}
