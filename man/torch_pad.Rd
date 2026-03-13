% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_pad}
\alias{torch_pad}
\title{Pad}
\usage{
torch_pad(self, pad, mode = "constant", value = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{pad}{An integer vector.}

\item{mode}{A character string.}

\item{value}{A numeric value. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::pad}.
}
