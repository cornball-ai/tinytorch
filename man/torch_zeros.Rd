% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_zeros}
\alias{torch_zeros}
\title{Create a tensor of zeros}
\usage{
torch_zeros(..., dtype = NULL, device = NULL)
}
\arguments{
\item{...}{Dimensions (as separate integers or a single integer vector).}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\description{
Create a tensor of zeros
}
\examples{
\donttest{
torch_zeros(c(2, 3))
}
}
