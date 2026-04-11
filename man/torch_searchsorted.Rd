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
\examples{
\donttest{
if (torch_is_installed()) {

sorted_sequence <- torch_tensor(rbind(c(1, 3, 5, 7, 9), c(2, 4, 6, 8, 10)))
sorted_sequence
values <- torch_tensor(rbind(c(3, 6, 9), c(3, 6, 9)))
values
torch_searchsorted(sorted_sequence, values)
torch_searchsorted(sorted_sequence, values, right=TRUE)
sorted_sequence_1d <- torch_tensor(c(1, 3, 5, 7, 9))
sorted_sequence_1d
torch_searchsorted(sorted_sequence_1d, values)
}
}
}
