% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_is_pinned}
\alias{torch_is_pinned}
\title{Is pinned}
\usage{
torch_is_pinned(self, device = NULL)
}
\arguments{
\item{self}{A torch_tensor.}

\item{device}{A device string (e.g., "cpu"). Optional.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::is_pinned}.
}
