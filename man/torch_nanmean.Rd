% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_nanmean}
\alias{torch_nanmean}
\title{Nanmean}
\usage{
torch_nanmean(self, dim = NULL, keepdim = FALSE, dtype = NULL)
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
Wrapper for \code{at::nanmean}.
}
