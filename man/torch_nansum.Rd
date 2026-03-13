% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nansum}
\alias{torch_nansum}
\title{Nansum}
\usage{
torch_nansum(self, dim = NULL, keepdim = FALSE, dtype = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{dim}{(int[1]). Optional.}

\item{keepdim}{A logical value.}

\item{dtype}{A torch dtype (e.g., torch_float32). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::nansum}.
}
