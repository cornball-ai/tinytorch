% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_ones}
\alias{torch_ones}
\title{Create a tensor of ones}
\usage{
torch_ones(..., dtype = NULL, device = NULL)
}
\arguments{
\item{...}{Dimensions (as separate integers or a single integer vector).}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\description{
Create a tensor of ones
}
\examples{
\dontrun{
torch_ones(c(2, 3))
}
}
