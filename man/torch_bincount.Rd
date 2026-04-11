% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_bincount}
\alias{torch_bincount}
\title{Bincount}
\usage{
torch_bincount(self, weights = NULL, minlength = 0)
}
\arguments{
\item{self}{A torch_tensor.}

\item{weights}{A torch_tensor. Optional.}

\item{minlength}{An integer.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::bincount}.
}
\examples{
\donttest{
if (torch_is_installed()) {

input = torch_randint(1, 8, list(5), dtype=torch_int64())
weights = torch_linspace(0, 1, steps=5)
input
weights
torch_bincount(input, weights)
input$bincount(weights)
}
}
}
