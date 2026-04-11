% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nanquantile}
\alias{torch_nanquantile}
\title{Nanquantile}
\usage{
torch_nanquantile(self, q, dim = NULL, keepdim = FALSE, interpolation = "linear")
}
\arguments{
\item{self}{A torch_tensor.}

\item{q}{A torch_tensor.}

\item{dim}{An integer. Optional.}

\item{keepdim}{A logical value.}

\item{interpolation}{A character string.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nanquantile}.
}
\examples{
\donttest{
if (torch_is_installed()) {

t <- torch_tensor(c(NaN, 1, 2))
t$quantile(0.5)
t$nanquantile(0.5)
t <- torch_tensor(rbind(c(NaN, NaN), c(1, 2)))
t
t$nanquantile(0.5, dim=1)
t$nanquantile(0.5, dim=2)
torch_nanquantile(t, 0.5, dim = 1)
torch_nanquantile(t, 0.5, dim = 2)
}
}
}
