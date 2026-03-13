% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_triangular_solve}
\alias{torch_triangular_solve}
\title{Triangular solve}
\usage{
torch_triangular_solve(self, A, upper = TRUE, transpose = FALSE,
                       unitriangular = FALSE)
}
\arguments{
\item{self}{A torch_tensor.}

\item{A}{A torch_tensor.}

\item{upper}{A logical value.}

\item{transpose}{A logical value.}

\item{unitriangular}{A logical value.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::triangular_solve}.
}
