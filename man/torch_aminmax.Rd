% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_aminmax}
\alias{torch_aminmax}
\title{Aminmax}
\usage{
torch_aminmax(self, dim = NULL, keepdim = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{An integer. Optional.}

\item{keepdim}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::aminmax}.
}
