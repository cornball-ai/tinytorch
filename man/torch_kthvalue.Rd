% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_kthvalue}
\alias{torch_kthvalue}
\title{Kthvalue}
\usage{
torch_kthvalue(self, k, dim = -1, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{k}{An integer.}

\item{dim}{An integer.}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::kthvalue}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_arange(1, 6)
x
torch_kthvalue(x, 4)
x <- torch_arange(1,6)$resize_(c(2,3))
x
torch_kthvalue(x, 2, 1, TRUE)
}
}
}
