% tinyrox says don't edit this manually, but it can't stop you!
\name{torch_movedim}
\alias{torch_movedim}
\title{Movedim}
\usage{
torch_movedim(self, source, destination)
}
\arguments{
\item{self}{A torch_tensor.}

\item{source}{An integer vector.}

\item{destination}{An integer vector.}
}
\value{
A torch_tensor.
}
\description{
Wrapper for \code{at::movedim}.
}
