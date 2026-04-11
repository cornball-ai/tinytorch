% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_exp2}
\alias{torch_exp2}
\title{Exp2}
\usage{
torch_exp2(self)
}
\arguments{
\item{self}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::exp2}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_exp2(torch_tensor(c(0, log2(2.), 3, 4)))
}
}
}
