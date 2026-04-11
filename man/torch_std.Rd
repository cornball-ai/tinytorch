% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_std}
\alias{torch_std}
\title{Standard deviation of a tensor}
\usage{
torch_std(self, dim = NULL, keepdim = FALSE, unbiased = TRUE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{Optional dimension to reduce.}

\item{keepdim}{Whether to keep the reduced dimension.}

\item{unbiased}{Logical. Use Bessel's correction. Default TRUE.}
}
\value{
A torch_tensor.
}
\description{
Standard deviation of a tensor
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(1, 3))
a
torch_std(a)


a = torch_randn(c(4, 4))
a
torch_std(a, dim=1)
}
}
}
