% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nansum}
\alias{torch_nansum}
\title{Nansum}
\usage{
torch_nansum(self, dim = NULL, keepdim = FALSE, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]). Optional.}

\item{keepdim}{A logical value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nansum}.
}
\examples{
\donttest{
if (torch_is_installed()) {

a <- torch_tensor(c(1., 2., NaN, 4.))
torch_nansum(a)


torch_nansum(torch_tensor(c(1., NaN)))
a <- torch_tensor(rbind(c(1, 2), c(3., NaN)))
torch_nansum(a)
torch_nansum(a, dim=1)
torch_nansum(a, dim=2)
}
}
}
