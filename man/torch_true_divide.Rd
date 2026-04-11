% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_true_divide}
\alias{torch_true_divide}
\title{True divide}
\usage{
torch_true_divide(self, other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::true_divide}.
}
\examples{
\donttest{
if (torch_is_installed()) {

dividend = torch_tensor(c(5, 3), dtype=torch_int())
divisor = torch_tensor(c(3, 2), dtype=torch_int())
torch_true_divide(dividend, divisor)
torch_true_divide(dividend, 2)
}
}
}
