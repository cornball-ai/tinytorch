% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_rand}
\alias{torch_rand}
\title{Rand}
\usage{
torch_rand(size, names, dtype = NULL, device = NULL)
}
\arguments{
\item{size}{An integer vector.}

\item{names}{A character vector of dimension names.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::rand}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_rand(4)
torch_rand(c(2, 3))
}
}
}
