% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_vander}
\alias{torch_vander}
\title{Vander}
\usage{
torch_vander(x, N = NULL, increasing = FALSE)
}
\arguments{
\item{x}{A torch_tensor.}

\item{N}{An integer. Optional.}

\item{increasing}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::vander}.
}
\examples{
\donttest{
if (torch_is_installed()) {

x <- torch_tensor(c(1, 2, 3, 5))
torch_vander(x)
torch_vander(x, N=3)
torch_vander(x, N=3, increasing=TRUE)
}
}
}
