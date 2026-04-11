% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bernoulli}
\alias{torch_bernoulli}
\title{Bernoulli}
\usage{
torch_bernoulli(self, generator = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{generator}{A random number generator. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bernoulli}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_empty(c(3, 3))$uniform_(0, 1)  # generate a uniform random matrix with range c(0, 1)
a
torch_bernoulli(a)
a = torch_ones(c(3, 3)) # probability of drawing "1" is 1
torch_bernoulli(a)
a = torch_zeros(c(3, 3)) # probability of drawing "1" is 0
torch_bernoulli(a)
}
}
}
