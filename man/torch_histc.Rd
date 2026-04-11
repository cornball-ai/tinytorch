% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_histc}
\alias{torch_histc}
\title{Histc}
\usage{
torch_histc(self, bins = 100, min = 0, max = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{bins}{An integer.}

\item{min}{A numeric scalar.}

\item{max}{A numeric scalar.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::histc}.
}
\examples{
\donttest{
if (torch_is_installed()) {

torch_histc(torch_tensor(c(1., 2, 1)), bins=4, min=0, max=3)
}
}
}
