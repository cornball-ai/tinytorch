% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_argmin}
\alias{torch_argmin}
\title{torch_argmin}
\usage{
torch_argmin(self, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A `torch_tensor`.}

\item{dim}{Integer dimension.}

\item{keepdim}{Logical; keep reduced dimensions.}
}
\value{
A `torch_tensor`.
}
\description{
torch_argmin
}
\examples{
\donttest{
if (torch_is_installed()) {

a = torch_randn(c(4, 4))
a
torch_argmin(a)


a = torch_randn(c(4, 4))
a
torch_argmin(a, dim=1)
}
}
}
