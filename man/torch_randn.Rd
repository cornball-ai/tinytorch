% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_randn}
\alias{torch_randn}
\title{Create a tensor with random normal values}
\usage{
torch_randn(..., dtype = NULL, device = NULL)
}
\arguments{
\item{...}{Dimensions (as separate integers or a single integer vector).}

\item{dtype}{Optional torch dtype.}

\item{device}{Ignored (CPU only).}
}
\value{
A `torch_tensor`.
}
\description{
Create a tensor with random normal values
}
\examples{
\donttest{
if (torch_is_installed()) {
torch_randn(c(2, 3))
}
}
}
