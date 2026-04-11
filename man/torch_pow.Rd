% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pow}
\alias{torch_pow}
\title{Power}
\usage{
torch_pow(self, exponent)
}
\arguments{
\item{self}{A torch_tensor or numeric scalar.}

\item{exponent}{A torch_tensor or numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Power
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4))
a
torch_pow(a, 2)
exp <- torch_arange(1, 5)
a <- torch_arange(1, 5)
a
exp
torch_pow(a, exp)


exp <- torch_arange(1, 5)
base <- 2
torch_pow(base, exp)
}
}
}
