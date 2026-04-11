% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_tensordot}
\alias{torch_tensordot}
\title{Tensordot}
\usage{
torch_tensordot(self, other, dims_self, dims_other)
}
\arguments{
\item{self}{A torch_tensor.}

\item{other}{A torch_tensor.}

\item{dims_self}{An integer vector.}

\item{dims_other}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::tensordot}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_arange(start = 1, end = 60)$reshape(c(3, 4, 5))
b <- torch_arange(start = 1, end = 24)$reshape(c(4, 3, 2))
torch_tensordot(a, b, dims = list(c(2, 1), c(1, 2)))

a = torch_randn(3, 4, 5, device='cuda')
b = torch_randn(4, 5, 6, device='cuda')
c = torch_tensordot(a, b, dims=2)$cpu()

}
}
}
