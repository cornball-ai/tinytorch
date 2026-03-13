% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_searchsorted}
\alias{torch_searchsorted}
\title{Searchsorted}
\usage{
torch_searchsorted(sorted_sequence, self, out_int32 = FALSE, right = FALSE,
                   side = NULL, sorter = NULL)
}
\arguments{
\item{sorted_sequence}{A torch_tensor.}

\item{self}{A torch_tensor.}

\item{out_int32}{A logical value.}

\item{right}{A logical value.}

\item{side}{A character string. Optional.}

\item{sorter}{A torch_tensor. Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::searchsorted}.
}
