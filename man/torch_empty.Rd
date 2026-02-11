% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_empty}
\alias{torch_empty}
\title{Create an uninitialized tensor}
\usage{
torch_empty(..., dtype = NULL, device = NULL)
}
\arguments{
\item{...}{Dimensions (as separate integers or a single integer vector).}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\description{
Create an uninitialized tensor
}
\examples{
\donttest{
torch_empty(c(2, 3))
}
}
