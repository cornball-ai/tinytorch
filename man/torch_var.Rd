% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_var}
\alias{torch_var}
\title{Var}
\usage{
torch_var(self, unbiased = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{unbiased}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::var}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_var(a)


a = torch_randn(c(4, 4))
a
torch_var(a, 1)
}
}
}
